"""Train and evaluate DirectionalThreatModel on the UGR16 feature mirror.

The output is a proxy-data experiment, not a raw-flow result. It is intended
to validate the end-to-end training path before the official raw CSV adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset, load_ugr16, log_standardize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="reports/ugr16_proxy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for current, history, target in loader:
            output = model(current.to(device), history.to(device))
            probabilities.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
            labels.append(target.numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    metrics: dict[str, float] = {
        "positive_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    np.random.seed(7)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split, official_test = load_ugr16(args.data_root)
    if len(np.unique(train_split.y_binary)) < 2 or len(np.unique(official_test.y_binary)) < 2:
        raise ValueError(
            "Binary UGR16 proxy training requires both normal and attack rows; "
            "use train_ugr16_severity for this all-positive proxy label set."
        )
    split_at = int(len(train_split.x) * 0.8)
    train_x, val_x = train_split.x[:split_at], train_split.x[split_at:]
    train_y, val_y = train_split.y_binary[:split_at], train_split.y_binary[split_at:]
    train_x, val_x, test_x = log_standardize(train_x, val_x, official_test.x)

    if args.max_train > 0:
        train_indices = np.arange(min(args.max_train, len(train_x)))
    else:
        train_indices = np.arange(len(train_x))
    train_ds = CausalSequenceDataset(train_x, train_y, args.history)
    val_ds = CausalSequenceDataset(val_x, val_y, args.history)
    test_ds = CausalSequenceDataset(test_x, official_test.y_binary, args.history)
    train_loader = DataLoader(Subset(train_ds, train_indices.tolist()), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionalThreatModel(input_dim=train_x.shape[1], hidden_dim=args.hidden_dim, classes=2).to(device)
    positive = max(float(train_y.sum()), 1.0)
    negative = max(float(len(train_y) - train_y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for current, sequence, target in train_loader:
            optimizer.zero_grad()
            output = model(current.to(device), sequence.to(device))
            loss = loss_fn(output["binary_logit"], target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, val_loader, device)
        epoch_record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_metrics}
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))

    test_metrics = evaluate(model, test_loader, device)
    results = {
        "dataset": "UGR16v2 feature mirror",
        "dataset_limitations": "One-minute aggregate feature data; not the original raw one-way NetFlow CSV release.",
        "feature_count": int(train_x.shape[1]),
        "history_length": args.history,
        "train_rows_used": int(len(train_indices)),
        "validation_rows": int(len(val_ds)),
        "test_rows": int(len(test_ds)),
        "device": str(device),
        "epochs": history,
        "test_metrics": test_metrics,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": vars(args), "results": results}, output_dir / "model.pt")
    print(json.dumps({"status": "complete", "test_metrics": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
