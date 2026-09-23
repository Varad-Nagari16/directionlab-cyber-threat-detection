"""Train the DirectionLab binary detector on prepared CIC-IDS2017 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset


def load_split(path: Path, feature_columns: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    if "label_binary" not in frame:
        raise ValueError(f"Missing label_binary in {path}")
    excluded = {"timestamp", "Label", "label_binary", "source_file", "Protocol"}
    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column not in excluded]
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing feature columns in {path}: {missing}")
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0).to_numpy(dtype=np.float32)
    y = frame["label_binary"].to_numpy(dtype=np.float32)
    return x, y, feature_columns


def transform(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    scaler = StandardScaler()
    train_log = np.log1p(train_x)
    output = [scaler.fit_transform(train_log).astype(np.float32)]
    for array in others:
        output.append(scaler.transform(np.log1p(array)).astype(np.float32))
    return tuple(output)


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for current, history, target in loader:
            output = model(current.to(device), history.to(device))
            probabilities.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    y_true, y_prob = collect_predictions(model, loader, device)
    return metrics_at_threshold(y_true, y_prob, threshold)


def calibrate_threshold(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    y_true, y_prob = collect_predictions(model, loader, device)
    best_threshold, best_score = 0.5, -1.0
    # Model scores can all be below 0.01 under strong distribution shift.
    # Search score quantiles and exact observed scores instead of assuming a
    # well-calibrated probability scale.
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 501)))
    for threshold in candidates:
        score = matthews_corrcoef(y_true, (y_prob >= threshold).astype(np.int64))
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, metrics_at_threshold(y_true, y_prob, best_threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/cicids2017")
    parser.add_argument("--output-dir", default="reports/cicids_binary")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=300000, help="0 uses every training row")
    parser.add_argument("--max-eval", type=int, default=100000, help="0 evaluates every row")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    root, output_dir = Path(args.data_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, feature_columns = load_split(root / "train.csv")
    val_x, val_y, _ = load_split(root / "validation.csv", feature_columns)
    calibration_path = root / "calibration.csv"
    calibration_x = calibration_y = None
    if calibration_path.exists():
        calibration_x, calibration_y, _ = load_split(calibration_path, feature_columns)
    test_x, test_y, _ = load_split(root / "test.csv", feature_columns)
    transformed = transform(train_x, val_x, *( [calibration_x] if calibration_x is not None else []), test_x)
    train_x, val_x = transformed[0], transformed[1]
    if calibration_x is not None:
        calibration_x = transformed[2]
        test_x = transformed[3]
    else:
        test_x = transformed[2]

    train_ds = CausalSequenceDataset(train_x, train_y, args.history)
    val_ds = CausalSequenceDataset(val_x, val_y, args.history)
    calibration_ds = CausalSequenceDataset(calibration_x, calibration_y, args.history) if calibration_x is not None else None
    test_ds = CausalSequenceDataset(test_x, test_y, args.history)
    train_count = len(train_ds) if args.max_train == 0 else min(args.max_train, len(train_ds))
    eval_count = lambda size: size if args.max_eval == 0 else min(args.max_eval, size)
    train_loader = DataLoader(Subset(train_ds, range(train_count)), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(Subset(val_ds, range(eval_count(len(val_ds)))), batch_size=args.batch_size, shuffle=False)
    calibration_loader = DataLoader(Subset(calibration_ds, range(eval_count(len(calibration_ds)))), batch_size=args.batch_size, shuffle=False) if calibration_ds is not None else None
    test_loader = DataLoader(Subset(test_ds, range(eval_count(len(test_ds)))), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionalThreatModel(input_dim=train_x.shape[1], hidden_dim=args.hidden_dim, classes=2).to(device)
    positive = max(float(train_y[:train_count].sum()), 1.0)
    negative = max(float(train_count - train_y[:train_count].sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    epochs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for current, history, target in train_loader:
            optimizer.zero_grad()
            output = model(current.to(device), history.to(device))
            loss = loss_fn(output["binary_logit"], target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": evaluate(model, val_loader, device)}
        epochs.append(record)
        print(json.dumps(record, sort_keys=True))

    threshold_loader = calibration_loader if calibration_loader is not None else val_loader
    calibrated_threshold, threshold_calibrated_metrics = calibrate_threshold(model, threshold_loader, device)
    test_metrics = evaluate(model, test_loader, device, calibrated_threshold)
    results = {
        "dataset": "CIC-IDS2017 MachineLearningCVE",
        "task": "binary benign-vs-attack classification",
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "history_length": args.history,
        "train_rows_used": train_count,
        "validation_rows_evaluated": eval_count(len(val_ds)),
        "calibration_rows_evaluated": eval_count(len(calibration_ds)) if calibration_ds is not None else 0,
        "test_rows_evaluated": eval_count(len(test_ds)),
        "device": str(device),
        "epochs": epochs,
        "test_metrics": test_metrics,
        "calibrated_threshold": calibrated_threshold,
        "threshold_calibration_metrics": threshold_calibrated_metrics,
        "directionality_caveat": "CICFlowMeter flow aggregates may summarize both directions; this is a binary flow-based prototype, not proof of strict one-way deployment validity.",
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": vars(args), "feature_columns": feature_columns, "results": results}, output_dir / "model.pt")
    print(json.dumps({"status": "complete", "test_metrics": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
