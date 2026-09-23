"""Create a separate stratified multiclass split with every family in training.

This is not a temporal/unseen-day evaluation. It measures family recognition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cicids2017")
    parser.add_argument("--output", default="data/processed/cicids2017_family")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(root / name, low_memory=False) for name in ("train.csv", "validation.csv", "test.csv")]
    combined = pd.concat(frames, ignore_index=True)
    combined["family_class"] = combined["Label"].map(family)
    train_parts, calibration_parts, test_parts = [], [], []
    for name, group in combined.groupby("family_class", sort=True):
        group = group.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n = len(group)
        train_end, calibration_end = int(n * 0.80), int(n * 0.90)
        train_parts.append(group.iloc[:train_end])
        calibration_parts.append(group.iloc[train_end:calibration_end])
        test_parts.append(group.iloc[calibration_end:])
    splits = {
        "train": pd.concat(train_parts).sample(frac=1.0, random_state=args.seed),
        "calibration": pd.concat(calibration_parts).sample(frac=1.0, random_state=args.seed),
        "test": pd.concat(test_parts).sample(frac=1.0, random_state=args.seed),
    }
    summary = {}
    for name, frame in splits.items():
        frame = frame.drop(columns=["family_class"])
        frame.to_csv(output / f"{name}.csv", index=False)
        summary[name] = {"rows": int(len(frame)), "families": frame["Label"].map(family).value_counts().to_dict()}
        print({name: summary[name]})
    manifest = {"policy": "random stratified 80/10/10 by attack family; not temporal", "summary": summary}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(output), "summary": summary}))


if __name__ == "__main__":
    main()
