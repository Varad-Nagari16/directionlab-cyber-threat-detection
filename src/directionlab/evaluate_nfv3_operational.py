"""Evaluate operational false alerts for NetFlow v3 host-held-out folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset

EXCLUDED = {
    "timestamp", "Label", "label_binary", "source_file", "source_host_group",
    "source_host_id", "Protocol", "Attack"
}


def load_frame(path: Path, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    if "label_binary" not in frame:
        raise ValueError(f"Missing label_binary in {path}")
    if feature_columns is None:
        feature_columns = [c for c in frame.columns if c not in EXCLUDED]
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0).to_numpy(dtype=np.float32)
    labels = frame["label_binary"].to_numpy(dtype=np.float32)
    return frame, values, feature_columns


def infer(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, history: int, batch_size: int, device: torch.device, limit: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = CausalSequenceDataset(x, y, history)
    count = len(dataset) if limit == 0 else min(limit, len(dataset))
    loader = DataLoader(Subset(dataset, range(count)), batch_size=batch_size, shuffle=False)
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for current, sequence, target in loader:
            output = model(current.to(device), sequence.to(device))
            scores.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", default="data/processed/nfv3_host_folds")
    parser.add_argument("--reports-root", default="reports/nfv3_cv")
    parser.add_argument("--output", default="reports/nfv3_cv/operational_metrics.csv")
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every test row")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, object]] = []
    for fold_dir in sorted(Path(args.fold_root).glob("fold_*")):
        checkpoint_path = Path(args.reports_root) / fold_dir.name / "model.pt"
        test_path = fold_dir / "test.csv"
        if not checkpoint_path.exists() or not test_path.exists():
            print(f"Skipping {fold_dir.name}: missing model.pt or test.csv")
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        feature_columns = checkpoint["feature_columns"]
        train_frame, train_x, _ = load_frame(fold_dir / "train.csv", feature_columns)
        test_frame, test_x, _ = load_frame(test_path, feature_columns)
        calibration_frame, calibration_x, _ = load_frame(fold_dir / "calibration.csv", feature_columns)
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
        scaler.n_features_in_ = len(feature_columns)
        train_scaled = scaler.transform(np.log1p(train_x)).astype(np.float32)
        test_scaled = scaler.transform(np.log1p(test_x)).astype(np.float32)
        config = checkpoint.get("config", {})
        hidden_dim = int(config.get("hidden_dim", 32))
        model = DirectionalThreatModel(input_dim=len(feature_columns), hidden_dim=hidden_dim, classes=2).to(device)
        model.load_state_dict(checkpoint["model"])
        threshold = float(checkpoint["results"]["calibrated_threshold"])
        scores, labels = infer(model, test_scaled, test_frame["label_binary"].to_numpy(dtype=np.float32), args.history, args.batch_size, device, args.limit)
        predicted = scores >= threshold
        benign = labels == 0
        attack = labels == 1
        false_alerts = int((predicted & benign).sum())
        true_negatives = int((~predicted & benign).sum())
        true_positives = int((predicted & attack).sum())
        false_negatives = int((~predicted & attack).sum())
        observed = test_frame.iloc[:len(labels)]
        start = pd.to_numeric(observed.get("FLOW_START_MILLISECONDS", pd.Series(dtype=float)), errors="coerce")
        end = pd.to_numeric(observed.get("FLOW_END_MILLISECONDS", pd.Series(dtype=float)), errors="coerce")
        valid_time = pd.concat([start, end], axis=1).dropna()
        duration_hours = float(max((valid_time.iloc[:, 1].max() - valid_time.iloc[:, 0].min()) / 3_600_000.0, 1e-9)) if not valid_time.empty else None
        records.append({
            "fold": fold_dir.name,
            "rows_evaluated": int(len(labels)),
            "benign_rows": int(benign.sum()),
            "attack_rows": int(attack.sum()),
            "false_alerts": false_alerts,
            "true_negatives": true_negatives,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "benign_false_positive_rate": false_alerts / max(int(benign.sum()), 1),
            "attack_recall": true_positives / max(int(attack.sum()), 1),
            "threshold": threshold,
            "observed_duration_hours": duration_hours,
            "false_alerts_per_hour": false_alerts / duration_hours if duration_hours else None,
            "note": "False alerts per hour uses the observed FLOW_START/FLOW_END timestamp span in the test file; it is not a packet arrival-rate estimate.",
        })
        print(json.dumps(records[-1], sort_keys=True))
    if not records:
        raise SystemExit("No fold checkpoints found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    Path(output.with_suffix(".json")).write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"status": "complete", "folds": len(records), "output": str(output)}))


if __name__ == "__main__":
    main()
