"""Train the causal model as an anomaly-severity forecaster on UGR16 proxy data.

The proxy contains a nonzero blacklist count for every observation, so the
valid target here is the intensity of that count, not a binary attack label.
Targets are log1p(blacklist_count). This is not a substitute for raw-flow
binary intrusion detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Subset

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset, add_time_features, load_ugr16, log_standardize


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for current, sequence, target in loader:
            output = model(current.to(device), sequence.to(device))
            predictions.append(output["severity"].cpu().numpy())
            targets.append(target.numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    baseline = np.full_like(y_true, y_true.mean())
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "mae_log1p": float(np.mean(np.abs(y_true - y_pred))),
        "rmse_log1p": rmse(y_true, y_pred),
        "baseline_rmse_log1p": rmse(y_true, baseline),
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "mean_true_log1p": float(y_true.mean()),
        "mean_pred_log1p": float(y_pred.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="reports/ugr16_severity")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--use-time-features", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_split, official_test = load_ugr16(args.data_root)

    split_at = int(len(train_split.x) * 0.8)
    train_x, val_x = train_split.x[:split_at], train_split.x[split_at:]
    train_target = np.log1p(train_split.attack_counts[:split_at, 4]).astype(np.float32)
    val_target = np.log1p(train_split.attack_counts[split_at:, 4]).astype(np.float32)
    test_target = np.log1p(official_test.attack_counts[:, 4]).astype(np.float32)
    train_x, val_x, test_x = log_standardize(train_x, val_x, official_test.x)
    if args.use_time_features:
        train_x = add_time_features(train_x, train_split.timestamps[:split_at])
        val_x = add_time_features(val_x, train_split.timestamps[split_at:])
        test_x = add_time_features(test_x, official_test.timestamps)

    train_ds = CausalSequenceDataset(train_x, train_target, args.history)
    val_ds = CausalSequenceDataset(val_x, val_target, args.history)
    test_ds = CausalSequenceDataset(test_x, test_target, args.history)
    train_count = min(args.max_train, len(train_ds)) if args.max_train > 0 else len(train_ds)
    indices = np.arange(train_count).tolist()
    train_loader = DataLoader(Subset(train_ds, indices), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionalThreatModel(input_dim=train_x.shape[1], hidden_dim=args.hidden_dim, classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    epochs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for current, sequence, target in train_loader:
            optimizer.zero_grad()
            output = model(current.to(device), sequence.to(device))
            loss = loss_fn(output["severity"], target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": evaluate(model, val_loader, device)}
        epochs.append(record)
        print(json.dumps(record, sort_keys=True))

    results = {
        "dataset": "UGR16v2 feature mirror",
        "task": "blacklist-volume severity forecasting",
        "target": "log1p(labelblacklist_count)",
        "dataset_limitations": "One-minute aggregate feature data; all rows have nonzero blacklist counts, so binary attack detection is invalid.",
        "feature_count": int(train_x.shape[1]),
        "history_length": args.history,
        "train_rows_used": int(train_count),
        "validation_rows": int(len(val_ds)),
        "test_rows": int(len(test_ds)),
        "device": str(device),
        "epochs": epochs,
        "test_metrics": evaluate(model, test_loader, device),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": vars(args), "results": results}, output_dir / "model.pt")
    print(json.dumps({"status": "complete", "test_metrics": results["test_metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
