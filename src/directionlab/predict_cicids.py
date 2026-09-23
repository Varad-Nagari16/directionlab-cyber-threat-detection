"""Run the final CIC binary detector and write JSONL alerts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="reports/cicids_binary_final/model.pt")
    parser.add_argument("--input", default="data/processed/cicids2017_final/test.csv")
    parser.add_argument("--output", default="reports/cicids_binary_final/alerts.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--history", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "scaler_mean" not in checkpoint:
        raise ValueError("Checkpoint lacks scaler parameters; retrain with the updated trainer first")
    frame = pd.read_csv(args.input, low_memory=False)
    columns = checkpoint["feature_columns"]
    x = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(np.float32)
    x = (np.log1p(x) - np.asarray(checkpoint["scaler_mean"], dtype=np.float32)) / np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    target = frame["label_binary"].to_numpy(np.float32) if "label_binary" in frame else np.zeros(len(frame), dtype=np.float32)
    dataset = CausalSequenceDataset(x, target, args.history)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    state = checkpoint["model"]
    hidden_dim = state["static_encoder.0.weight"].shape[0]
    model = DirectionalThreatModel(input_dim=len(columns), hidden_dim=hidden_dim, classes=2)
    model.load_state_dict(state)
    model.eval()
    threshold = float(checkpoint["results"]["calibrated_threshold"])
    alerts, offset = [], 0
    with torch.no_grad():
        for current, history, _ in loader:
            scores = torch.sigmoid(model(current, history)["binary_logit"]).numpy()
            for score in scores:
                if len(alerts) >= args.limit:
                    break
                source = frame.iloc[offset]
                alert = {
                    "observation_index": int(offset),
                    "source_file": str(source.get("source_file", "unknown")),
                    "threat_class": "attack" if float(score) >= threshold else "benign",
                    "attack_score": float(score),
                    "threshold": threshold,
                    "severity": "HIGH" if float(score) >= threshold else "LOW",
                }
                if "Label" in frame:
                    alert["reference_label"] = str(source["Label"])
                alerts.append(alert)
                offset += 1
            if len(alerts) >= args.limit:
                break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(alert) for alert in alerts) + "\n")
    print(json.dumps({"status": "ok", "alerts_written": len(alerts), "threshold": threshold, "output": str(output)}))


if __name__ == "__main__":
    main()
