"""Create realistic day-based CIC-IDS2017 splits from prepared CSV files.

Train: Monday + Tuesday
Validation: Wednesday
Calibration: Thursday
Test: Friday
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def day_of(source_file: str) -> str:
    lowered = source_file.lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        if day in lowered:
            return day.title()
    return "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cicids2017")
    parser.add_argument("--output", default="data/processed/cicids2017_day")
    args = parser.parse_args()
    input_root, output_root = Path(args.input), Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(input_root / name, low_memory=False) for name in ("train.csv", "validation.csv", "test.csv")]
    combined = pd.concat(frames, ignore_index=True)
    combined["capture_day"] = combined["source_file"].map(day_of)
    if "Unknown" in set(combined["capture_day"]):
        raise ValueError("Could not determine capture day from source_file")

    splits = {
        "train": combined[combined["capture_day"].isin(["Monday", "Tuesday"])],
        "validation": combined[combined["capture_day"] == "Wednesday"],
        "calibration": combined[combined["capture_day"] == "Thursday"],
        "test": combined[combined["capture_day"] == "Friday"],
    }
    summary = {}
    for name, frame in splits.items():
        frame = frame.drop(columns=["capture_day"])
        frame.to_csv(output_root / f"{name}.csv", index=False)
        summary[name] = {"rows": len(frame), "attack_rate": float(frame["label_binary"].mean())}
        print({name: summary[name]})

    manifest = {
        "split_policy": "train Monday+Tuesday; validation Wednesday; calibration Thursday; test Friday",
        "summary": summary,
        "source_files": sorted(combined["source_file"].unique().tolist()),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(output_root), "summary": summary}))


if __name__ == "__main__":
    main()
