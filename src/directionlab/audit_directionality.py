"""Audit prepared CIC-IDS2017 data and checkpoints for directionality leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FORBIDDEN_TOKENS = (
    "backward",
    "bwd",
    "payload",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "source_ip",
    "destination_ip",
    "src_ip",
    "dst_ip",
)


def forbidden(name: str) -> bool:
    lowered = name.lower().replace("-", " ")
    return any(token in lowered for token in FORBIDDEN_TOKENS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="reports/cicids_binary_final/model.pt")
    parser.add_argument("--manifest", default="data/processed/cicids2017_final/manifest.json")
    parser.add_argument("--output", default="reports/directionality_audit.json")
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = list(checkpoint.get("feature_columns", []))
    forbidden_features = [name for name in features if forbidden(name)]
    manifest = json.loads(Path(args.manifest).read_text()) if Path(args.manifest).exists() else {}
    result = {
        "checkpoint": args.checkpoint,
        "feature_count": len(features),
        "feature_columns": features,
        "forbidden_feature_columns": forbidden_features,
        "passed_feature_audit": not forbidden_features,
        "manifest_directionality_caveat": manifest.get("directionality_caveat", "not recorded"),
        "interpretation": "The checkpoint contains no explicit reverse-flow or payload feature names. This audit cannot prove that the source dataset was collected from a physically one-way sensor.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if forbidden_features:
        raise SystemExit("Directionality audit failed: forbidden feature names found")


if __name__ == "__main__":
    main()
