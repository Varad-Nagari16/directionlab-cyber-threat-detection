"""Run the family-balanced multiclass CIC detector and write JSONL alerts."""

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
    parser.add_argument("--checkpoint", default="reports/cicids_multiclass_family/model.pt")
    parser.add_argument("--input", default="data/processed/cicids2017_family/test.csv")
    parser.add_argument("--output", default="reports/cicids_multiclass_family/alerts.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--bot-threshold", type=float, default=None, help="Reject Bot argmax predictions below this calibrated score")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    columns = checkpoint["feature_columns"]
    frame = pd.read_csv(args.input, low_memory=False)
    x = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(np.float32)
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    x = (np.log1p(x) - mean) / scale
    target = np.zeros(len(frame), dtype=np.float32)
    dataset = CausalSequenceDataset(x, target, args.history)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    state = checkpoint["model"]
    hidden_dim = state["static_encoder.0.weight"].shape[0]
    model = DirectionalThreatModel(input_dim=len(columns), hidden_dim=hidden_dim, classes=len(classes))
    model.load_state_dict(state)
    model.eval()

    alerts, offset = [], 0
    with torch.no_grad():
        for current, history, _ in loader:
            probabilities = torch.softmax(model(current, history)["multiclass_logits"], dim=1).numpy()
            for scores in probabilities:
                if len(alerts) >= args.limit:
                    break
                predicted_id = int(np.argmax(scores))
                if args.bot_threshold is not None and classes[predicted_id] == "Bot" and float(scores[predicted_id]) < args.bot_threshold:
                    alternatives = [i for i, name in enumerate(classes) if name != "Bot"]
                    predicted_id = max(alternatives, key=lambda i: scores[i])
                source = frame.iloc[offset]
                alert = {
                    "observation_index": int(offset),
                    "source_file": str(source.get("source_file", "unknown")),
                    "threat_class": classes[predicted_id],
                    "class_scores": {classes[i]: float(scores[i]) for i in range(len(classes))},
                    "attack_score": float(1.0 - scores[0]),
                    "severity": "LOW" if predicted_id == 0 else ("CRITICAL" if float(scores[predicted_id]) >= 0.9 else "HIGH"),
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
    print(json.dumps({"status": "ok", "alerts_written": len(alerts), "classes": classes, "output": str(output)}))


if __name__ == "__main__":
    main()
