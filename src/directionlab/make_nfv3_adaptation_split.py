"""Create temporal adaptation and later-evaluation files without label filtering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def split_file(path: Path, output: Path, fraction: float) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    if "FLOW_START_MILLISECONDS" not in frame:
        raise ValueError(f"Missing FLOW_START_MILLISECONDS in {path}")
    frame["__sort_time"] = pd.to_numeric(frame["FLOW_START_MILLISECONDS"], errors="coerce")
    frame = frame.sort_values(["__sort_time"], kind="stable").drop(columns=["__sort_time"]).reset_index(drop=True)
    cut = max(1, min(len(frame) - 1, int(len(frame) * fraction)))
    adaptation = frame.iloc[:cut].copy()
    evaluation = frame.iloc[cut:].copy()
    output.mkdir(parents=True, exist_ok=True)
    adaptation.to_csv(output / "adaptation.csv", index=False)
    evaluation.to_csv(output / "evaluation.csv", index=False)
    audit = {
        "source": str(path),
        "rows_total": int(len(frame)),
        "adaptation_rows": int(len(adaptation)),
        "evaluation_rows": int(len(evaluation)),
        "adaptation_start_ms": float(pd.to_numeric(adaptation["FLOW_START_MILLISECONDS"], errors="coerce").min()),
        "adaptation_end_ms": float(pd.to_numeric(adaptation["FLOW_END_MILLISECONDS"], errors="coerce").max()),
        "evaluation_start_ms": float(pd.to_numeric(evaluation["FLOW_START_MILLISECONDS"], errors="coerce").min()),
        "evaluation_end_ms": float(pd.to_numeric(evaluation["FLOW_END_MILLISECONDS"], errors="coerce").max()),
        "audit_attack_rate_adaptation": float(adaptation["label_binary"].mean()) if "label_binary" in adaptation else None,
        "audit_attack_rate_evaluation": float(evaluation["label_binary"].mean()) if "label_binary" in evaluation else None,
        "warning": "The adaptation split is selected by time only. Labels are recorded for audit and must not be used to filter or select the adaptation rows during deployment.",
    }
    (output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", default="data/processed/nfv3_full_host_folds")
    parser.add_argument("--output-root", default="data/processed/nfv3_adaptation")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--adaptation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.adaptation_fraction < 1:
        raise ValueError("--adaptation-fraction must be between 0 and 1")
    fold_dir = Path(args.fold_root) / f"fold_{args.fold:02d}"
    result = split_file(fold_dir / "test.csv", Path(args.output_root) / f"fold_{args.fold:02d}", args.adaptation_fraction)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
