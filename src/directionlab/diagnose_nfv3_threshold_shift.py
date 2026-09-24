"""Diagnose score-distribution shift across NetFlow v3 host-held-out folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset

EXCLUDED = {"timestamp", "Label", "label_binary", "source_file", "source_host_group", "source_host_id", "Protocol", "Attack"}
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def load(path: Path, features: list[str]):
    frame = pd.read_csv(path, low_memory=False)
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
    y = frame["label_binary"].to_numpy(dtype=np.float32)
    return frame, x, y


def scores(model, x, y, history, batch_size, device):
    loader = DataLoader(CausalSequenceDataset(x, y, history), batch_size=batch_size, shuffle=False)
    result = []
    model.eval()
    with torch.no_grad():
        for current, sequence, _ in loader:
            out = model(current.to(device), sequence.to(device))
            result.append(torch.sigmoid(out["binary_logit"]).cpu().numpy())
    return np.concatenate(result)


def describe(values: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "above_global_threshold": float((values >= threshold).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", default="data/processed/nfv3_host_folds")
    parser.add_argument("--reports-root", default="reports/nfv3_cv")
    parser.add_argument("--global-threshold", type=float, default=0.948996901512146)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", default="reports/nfv3_cv/threshold_shift_diagnostics.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for fold_dir in sorted(Path(args.fold_root).glob("fold_*")):
        checkpoint_path = Path(args.reports_root) / fold_dir.name / "model.pt"
        if not checkpoint_path.exists():
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        features = checkpoint["feature_columns"]
        cal_frame, cal_x, cal_y = load(fold_dir / "calibration.csv", features)
        test_frame, test_x, test_y = load(fold_dir / "test.csv", features)
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
        scaler.n_features_in_ = len(features)
        cal_x = scaler.transform(np.log1p(cal_x)).astype(np.float32)
        test_x = scaler.transform(np.log1p(test_x)).astype(np.float32)
        config = checkpoint.get("config", {})
        model = DirectionalThreatModel(input_dim=len(features), hidden_dim=int(config.get("hidden_dim", 32)), classes=2).to(device)
        model.load_state_dict(checkpoint["model"])
        cal_score = scores(model, cal_x, cal_y, args.history, args.batch_size, device)
        test_score = scores(model, test_x, test_y, args.history, args.batch_size, device)
        record = {
            "fold": fold_dir.name,
            "global_threshold": args.global_threshold,
            "calibration": {
                "benign": describe(cal_score[cal_y == 0], args.global_threshold),
                "attack": describe(cal_score[cal_y == 1], args.global_threshold),
            },
            "test": {
                "benign": describe(test_score[test_y == 0], args.global_threshold),
                "attack": describe(test_score[test_y == 1], args.global_threshold),
            },
        }
        results.append(record)
        print(json.dumps(record, sort_keys=True))
    if not results:
        raise SystemExit("No checkpoints found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": "complete", "folds": len(results), "output": str(output)}))


if __name__ == "__main__":
    main()
