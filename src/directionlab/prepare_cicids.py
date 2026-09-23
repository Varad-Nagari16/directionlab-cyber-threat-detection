"""Prepare CIC-IDS2017 CSVs for the direction-preserving detector.

This is a conservative prototype adapter. It drops identifiers, payload-like
fields, all backward-flow columns, and any unrecognized columns. Flow-level
features such as Flow Duration and Flow Bytes/s may still summarize a CIC flow
and must be described as a proxy for one-way observation until raw directional
records are available.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWLIST = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Average Packet Size",
]


def clean_name(name: object) -> str:
    value = str(name).replace("\ufeff", "").replace("\ufffd", "")
    value = value.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


def resolve_columns(columns: list[str]) -> dict[str, str]:
    """Resolve required names despite CICFlowMeter header variations."""
    normalized = {re.sub(r"[^a-z0-9]", "", column.lower()): column for column in columns}
    resolved = {}
    for wanted in ALLOWLIST + ["Label"]:
        key = re.sub(r"[^a-z0-9]", "", wanted.lower())
        if key in normalized:
            resolved[wanted] = normalized[key]
    return resolved


def read_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8", low_memory=False)
    frame.columns = [clean_name(column) for column in frame.columns]
    resolved = resolve_columns(list(frame.columns))
    missing = [column for column in ALLOWLIST + ["Label"] if column not in resolved]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}; actual columns: {list(frame.columns)}")
    frame = frame[[resolved[column] for column in ALLOWLIST + ["Label"]]].copy()
    frame.columns = ALLOWLIST + ["Label"]
    frame["Label"] = frame["Label"].astype(str).str.strip()
    frame["label_binary"] = (~frame["Label"].str.upper().eq("BENIGN")).astype(np.int8)
    # The MachineLearningCVE CSVs omit Timestamp and Protocol. Preserve their
    # capture row order as a monotonic observation index; do not fabricate a
    # wall-clock timestamp or protocol value.
    frame["timestamp"] = np.arange(len(frame), dtype=np.int64)
    for column in frame.columns:
        if column not in {"Label", "timestamp"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            frame[column] = frame[column].clip(lower=0)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/cicids2017")
    parser.add_argument("--output", default="data/processed/cicids2017")
    args = parser.parse_args()
    input_root, output_root = Path(args.input), Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    weekday_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
    files = sorted(input_root.rglob("*.csv"), key=lambda path: (min((rank for day, rank in weekday_order.items() if day.lower() in path.name.lower()), default=99), path.name))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_root}")

    frames = []
    for path in files:
        print(f"reading {path}")
        frame = read_file(path)
        frame["source_file"] = path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    combined["timestamp"] = np.arange(len(combined), dtype=np.int64)
    if combined["label_binary"].nunique() != 2:
        raise ValueError("Prepared data must contain both benign and attack rows")

    n = len(combined)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    splits = {
        "train": combined.iloc[:train_end],
        "validation": combined.iloc[train_end:val_end],
        "test": combined.iloc[val_end:],
    }
    for name, split in splits.items():
        split.to_csv(output_root / f"{name}.csv", index=False)
        print({name: {"rows": len(split), "attack_rate": float(split["label_binary"].mean())}})

    manifest = {
        "source_files": [str(path) for path in files],
        "rows": n,
        "allowlist": ALLOWLIST,
        "dropped_feature_policy": "identifiers, labels, payload-like columns, all backward-flow columns, and unrecognized columns",
        "chronological_splits": {name: len(split) for name, split in splits.items()},
        "label_counts": combined["label_binary"].value_counts().sort_index().to_dict(),
        "attack_families": combined["Label"].value_counts().to_dict(),
        "directionality_caveat": "CICFlowMeter flow-level aggregates may summarize both directions; this is a conservative binary prototype, not proof of strict one-way deployment validity.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=int) + "\n")
    print(json.dumps({"status": "ok", "output": str(output_root), "rows": n}))


if __name__ == "__main__":
    main()
