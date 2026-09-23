"""Train a balanced multiclass detector on the final CIC-IDS2017 day split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset

CLASS_NAMES = ["BENIGN", "DDoS", "PortScan", "Bot", "OTHER_ATTACK"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def family(label: object) -> str:
    value = str(label).strip().lower()
    if value == "benign":
        return "BENIGN"
    if value == "ddos":
        return "DDoS"
    if value == "portscan":
        return "PortScan"
    if value == "bot":
        return "Bot"
    return "OTHER_ATTACK"


def load_split(path: Path, feature_columns: list[str] | None = None):
    frame = pd.read_csv(path, low_memory=False)
    if feature_columns is None:
        excluded = {"timestamp", "Label", "label_binary", "source_file"}
        feature_columns = [column for column in frame.columns if column not in excluded]
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(np.float32)
    y = frame["Label"].map(family).map(CLASS_TO_ID).to_numpy(np.int64)
    return x, y, feature_columns


def transform(train_x: np.ndarray, *others: np.ndarray):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.log1p(train_x)).astype(np.float32)
    return (train_scaled, *[scaler.transform(np.log1p(array)).astype(np.float32) for array in others]), scaler


def evaluate(model, loader, device):
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for current, history, target in loader:
            logits = model(current.to(device), history.to(device))["multiclass_logits"]
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            truth.extend(target.numpy().astype(np.int64).tolist())
    report = classification_report(truth, predicted, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return {
        "rows": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "per_class": {name: {metric: float(values[metric]) for metric in ("precision", "recall", "f1-score", "support")} for name, values in report.items() if name in CLASS_NAMES},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/cicids2017_final")
    parser.add_argument("--output-dir", default="reports/cicids_multiclass_final")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    root, output_dir = Path(args.data_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, columns = load_split(root / "train.csv")
    calibration_x, calibration_y, _ = load_split(root / "calibration.csv", columns)
    test_x, test_y, _ = load_split(root / "test.csv", columns)
    transformed, scaler = transform(train_x, calibration_x, test_x)
    train_x, calibration_x, test_x = transformed

    train_ds = CausalSequenceDataset(train_x, train_y.astype(np.float32), args.history)
    calibration_ds = CausalSequenceDataset(calibration_x, calibration_y.astype(np.float32), args.history)
    test_ds = CausalSequenceDataset(test_x, test_y.astype(np.float32), args.history)
    train_indices = np.arange(len(train_ds)) if args.max_train == 0 or args.max_train >= len(train_ds) else np.linspace(0, len(train_ds) - 1, args.max_train, dtype=np.int64)
    train_labels = train_y[train_indices]
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Training subset must contain at least two classes")
    eval_count = lambda size: size if args.max_eval == 0 else min(size, args.max_eval)
    train_loader = DataLoader(Subset(train_ds, train_indices.tolist()), batch_size=args.batch_size, shuffle=True)
    calibration_loader = DataLoader(Subset(calibration_ds, range(eval_count(len(calibration_ds)))), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(test_ds, range(eval_count(len(test_ds)))), batch_size=args.batch_size)

    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    present = counts > 0
    weights[present] = len(train_labels) / (len(CLASS_NAMES) * counts[present])
    weights[~present] = 0.0
    weights = np.clip(weights, 0.25, 10.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionalThreatModel(input_dim=train_x.shape[1], hidden_dim=args.hidden_dim, classes=len(CLASS_NAMES)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for current, history, target in train_loader:
            optimizer.zero_grad()
            logits = model(current.to(device), history.to(device))["multiclass_logits"]
            loss = loss_fn(logits, target.long().to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, calibration_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "calibration": metrics}
        epochs.append(record)
        print(json.dumps(record))

    test_metrics = evaluate(model, test_loader, device)
    results = {
        "dataset": "CIC-IDS2017 MachineLearningCVE",
        "classes": CLASS_NAMES,
        "class_mapping": "BENIGN, DDoS, PortScan, Bot, and all remaining non-benign labels mapped to OTHER_ATTACK",
        "feature_columns": columns,
        "feature_count": len(columns),
        "history_length": args.history,
        "train_rows_used": int(len(train_indices)),
        "calibration_rows_evaluated": int(eval_count(len(calibration_ds))),
        "test_rows_evaluated": int(eval_count(len(test_ds))),
        "class_counts_train": {CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))},
        "class_weights": weights.tolist(),
        "epochs": epochs,
        "test_metrics": test_metrics,
        "directionality_caveat": "CICFlowMeter flow aggregates may summarize both directions; this is a multiclass flow-based prototype, not proof of strict one-way deployment validity.",
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": vars(args), "classes": CLASS_NAMES, "feature_columns": columns, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "results": results}, output_dir / "model.pt")
    print(json.dumps({"status": "complete", "test_metrics": test_metrics}))


if __name__ == "__main__":
    main()
