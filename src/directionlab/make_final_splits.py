"""Create final pre-deployment CIC-IDS2017 splits.

Train: Monday + Tuesday + Wednesday
Calibration: Thursday
Test: Friday
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def day_of(source_file: str) -> str:
    name = source_file.lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        if day in name:
            return day.title()
    return "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cicids2017")
    parser.add_argument("--output", default="data/processed/cicids2017_final")
    args = parser.parse_args()
    root, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(root / name, low_memory=False) for name in ("train.csv", "validation.csv", "test.csv")]
    combined = pd.concat(frames, ignore_index=True)
    combined["capture_day"] = combined["source_file"].map(day_of)
    if "Unknown" in set(combined["capture_day"]):
        raise ValueError("Could not determine capture day")
    splits = {
        "train": combined[combined["capture_day"].isin(["Monday", "Tuesday", "Wednesday"])],
        "calibration": combined[combined["capture_day"] == "Thursday"],
        "test": combined[combined["capture_day"] == "Friday"],
    }
    summary = {}
    for name, frame in splits.items():
        frame = frame.drop(columns=["capture_day"])
        frame.to_csv(output / f"{name}.csv", index=False)
        summary[name] = {"rows": int(len(frame)), "attack_rate": float(frame["label_binary"].mean())}
        print({name: summary[name]})
    (output / "manifest.json").write_text(json.dumps({"split_policy": "train Monday+Tuesday+Wednesday; calibration Thursday; test Friday", "summary": summary}, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(output), "summary": summary}))


if __name__ == "__main__":
    main()
