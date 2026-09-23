"""Evaluate a simple linear baseline on the same CIC-IDS2017 day split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def load(path: Path, columns: list[str] | None = None):
    frame = pd.read_csv(path, low_memory=False)
    excluded = {"timestamp", "Label", "label_binary", "source_file"}
    if columns is None:
        columns = [c for c in frame.columns if c not in excluded]
    x = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(np.float32)
    y = frame["label_binary"].to_numpy(np.int64)
    return x, y, columns


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(np.int64)
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "average_precision": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "threshold": float(threshold),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/cicids2017_day")
    parser.add_argument("--output", default="reports/cicids_logistic_baseline.json")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()
    root = Path(args.data_root)
    train_x, train_y, columns = load(root / "train.csv")
    calibration_x, calibration_y, _ = load(root / "calibration.csv", columns)
    test_x, test_y, _ = load(root / "test.csv", columns)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(np.log1p(train_x))
    calibration_x = scaler.transform(np.log1p(calibration_x))
    test_x = scaler.transform(np.log1p(test_x))
    model = LogisticRegression(class_weight="balanced", max_iter=200, solver="lbfgs", n_jobs=-1)
    model.fit(train_x, train_y)
    calibration_score = model.predict_proba(calibration_x)[:, 1]
    benign_scores = np.sort(calibration_score[calibration_y == 0])
    index = min(len(benign_scores) - 1, int(np.ceil((1 - args.target_fpr) * len(benign_scores))) - 1)
    threshold = float(benign_scores[max(index, 0)])
    test_score = model.predict_proba(test_x)[:, 1]
    result = {
        "model": "logistic_regression",
        "feature_count": len(columns),
        "target_fpr": args.target_fpr,
        "threshold": threshold,
        "calibration": metrics(calibration_y, calibration_score, threshold),
        "test": metrics(test_y, test_score, threshold),
        "feature_columns": columns,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
