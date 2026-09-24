"""Materialize full host-held-out folds from prepared NetFlow v3 data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def assign(frame: pd.DataFrame, fold: dict[str, object]) -> pd.Series:
    host = frame["source_host_id"].astype(str)
    group = frame["source_host_group"].astype(str)
    output = pd.Series("drop", index=frame.index, dtype="string")
    output.loc[host.isin(set(fold["train_attack_hosts"]))] = "train"
    output.loc[host == str(fold["calibration_attack_host"])] = "calibration"
    output.loc[host == str(fold["test_attack_host"])] = "test"
    output.loc[group.isin(set(fold["train_benign_groups"]))] = "train"
    output.loc[group.isin(set(fold["calibration_benign_groups"]))] = "calibration"
    output.loc[group.isin(set(fold["test_benign_groups"]))] = "test"
    return output


def materialize(data_root: Path, output_root: Path, fold: dict[str, object], number: int, chunksize: int) -> dict[str, object]:
    destination = output_root / f"fold_{number:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob("*.csv"):
        path.unlink()
    writers: dict[str, object] = {}
    rows = {name: 0 for name in ("train", "calibration", "test")}
    attacks = {name: 0 for name in rows}
    try:
        for source in (data_root / "train.csv", data_root / "calibration.csv", data_root / "test.csv"):
            for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
                required = {"source_host_id", "source_host_group", "label_binary"}
                missing = required.difference(chunk.columns)
                if missing:
                    raise ValueError(f"{source} missing columns: {sorted(missing)}")
                assignment = assign(chunk, fold)
                for name in rows:
                    part = chunk.loc[assignment == name]
                    if part.empty:
                        continue
                    path = destination / f"{name}.csv"
                    if name not in writers:
                        writers[name] = path.open("w", encoding="utf-8", newline="")
                        part.to_csv(writers[name], index=False)
                    else:
                        part.to_csv(writers[name], index=False, header=False)
                    rows[name] += len(part)
                    attacks[name] += int(part["label_binary"].sum())
    finally:
        for handle in writers.values():
            handle.close()
    summary = {name: {"rows": rows[name], "attacks": attacks[name], "attack_rate": attacks[name] / max(rows[name], 1)} for name in rows}
    manifest = {"fold_number": number, "fold": fold, "summary": summary, "policy": "attack hosts and benign source-host groups are both held out by fold"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/nfv3_directional")
    parser.add_argument("--plan", default="reports/nfv3_full_host_folds.json")
    parser.add_argument("--output-root", default="data/processed/nfv3_full_host_folds")
    parser.add_argument("--fold", type=int, default=0, help="1-based fold number; 0 materializes all")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    folds = plan["folds"]
    if args.fold < 0 or args.fold > len(folds):
        raise ValueError(f"--fold must be between 0 and {len(folds)}")
    selected = enumerate(folds, 1) if args.fold == 0 else [(args.fold, folds[args.fold - 1])]
    for number, fold in selected:
        result = materialize(Path(args.data_root), Path(args.output_root), fold, number, args.chunksize)
        print(json.dumps({"status": "ok", "fold": number, "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
