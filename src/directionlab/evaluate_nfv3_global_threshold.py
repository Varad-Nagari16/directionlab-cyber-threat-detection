"""Evaluate one global threshold selected from pooled calibration traffic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset

EXCLUDED = {"timestamp", "Label", "label_binary", "source_file", "source_host_group", "source_host_id", "Protocol", "Attack"}


def load_frame(path: Path, features: list[str] | None = None):
    frame = pd.read_csv(path, low_memory=False)
    if features is None:
        features = [c for c in frame.columns if c not in EXCLUDED]
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
    y = frame["label_binary"].to_numpy(dtype=np.float32)
    return frame, x, y, features


def score(model, x, y, history, batch_size, device):
    dataset = CausalSequenceDataset(x, y, history)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for current, sequence, target in loader:
            out = model(current.to(device), sequence.to(device))
            scores.append(torch.sigmoid(out["binary_logit"]).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def metrics(frame, scores, labels, threshold):
    predicted = scores >= threshold
    benign = labels == 0
    attack = labels == 1
    fp = int((predicted & benign).sum())
    tn = int((~predicted & benign).sum())
    tp = int((predicted & attack).sum())
    fn = int((~predicted & attack).sum())
    start = pd.to_numeric(frame.iloc[:len(labels)].get("FLOW_START_MILLISECONDS", pd.Series(dtype=float)), errors="coerce")
    end = pd.to_numeric(frame.iloc[:len(labels)].get("FLOW_END_MILLISECONDS", pd.Series(dtype=float)), errors="coerce")
    times = pd.concat([start, end], axis=1).dropna()
    hours = float(max((times.iloc[:, 1].max() - times.iloc[:, 0].min()) / 3_600_000.0, 1e-9)) if not times.empty else None
    return {
        "rows": int(len(labels)), "benign_rows": int(benign.sum()), "attack_rows": int(attack.sum()),
        "false_alerts": fp, "true_negatives": tn, "true_positives": tp, "false_negatives": fn,
        "benign_false_positive_rate": fp / max(int(benign.sum()), 1),
        "attack_recall": tp / max(int(attack.sum()), 1),
        "threshold": float(threshold), "observed_duration_hours": hours,
        "false_alerts_per_hour": fp / hours if hours else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", default="data/processed/nfv3_host_folds")
    parser.add_argument("--reports-root", default="reports/nfv3_cv")
    parser.add_argument("--output", default="reports/nfv3_cv/global_threshold_metrics.csv")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration_scores, calibration_labels = [], []
    test_records = []
    for fold_dir in sorted(Path(args.fold_root).glob("fold_*")):
        checkpoint_path = Path(args.reports_root) / fold_dir.name / "model.pt"
        if not checkpoint_path.exists():
            print(f"Skipping {fold_dir.name}: checkpoint missing")
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        features = checkpoint["feature_columns"]
        _, train_x, train_y, _ = load_frame(fold_dir / "train.csv", features)
        cal_frame, cal_x, cal_y, _ = load_frame(fold_dir / "calibration.csv", features)
        test_frame, test_x, test_y, _ = load_frame(fold_dir / "test.csv", features)
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
        scaler.n_features_in_ = len(features)
        cal_x = scaler.transform(np.log1p(cal_x)).astype(np.float32)
        test_x = scaler.transform(np.log1p(test_x)).astype(np.float32)
        config = checkpoint.get("config", {})
        model = DirectionalThreatModel(input_dim=len(features), hidden_dim=int(config.get("hidden_dim", 32)), classes=2).to(device)
        model.load_state_dict(checkpoint["model"])
        cal_scores, cal_labels = score(model, cal_x, cal_y, args.history, args.batch_size, device)
        test_scores, test_labels = score(model, test_x, test_y, args.history, args.batch_size, device)
        calibration_scores.append(cal_scores)
        calibration_labels.append(cal_labels)
        test_records.append((fold_dir.name, test_frame, test_scores, test_labels))
        print(f"Collected {fold_dir.name}: calibration={len(cal_scores)}, test={len(test_scores)}")
    if not test_records:
        raise SystemExit("No completed fold checkpoints found")
    pooled_scores = np.concatenate(calibration_scores)
    pooled_labels = np.concatenate(calibration_labels)
    benign_scores = np.sort(pooled_scores[pooled_labels == 0])
    index = min(len(benign_scores) - 1, int(np.ceil((1.0 - args.target_fpr) * len(benign_scores))) - 1)
    threshold = float(benign_scores[max(index, 0)])
    records = []
    for fold, frame, scores, labels in test_records:
        record = metrics(frame, scores, labels, threshold)
        record["fold"] = fold
        records.append(record)
        print(json.dumps(record, sort_keys=True))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    result = {"target_fpr": args.target_fpr, "global_threshold": threshold, "pooled_calibration_rows": int(len(pooled_labels)), "pooled_calibration_benign_rows": int((pooled_labels == 0).sum()), "folds": records, "note": "One threshold selected from pooled calibration scores and applied unchanged to every test host."}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "global_threshold": threshold, "output": str(out)}))


if __name__ == "__main__":
    main()
