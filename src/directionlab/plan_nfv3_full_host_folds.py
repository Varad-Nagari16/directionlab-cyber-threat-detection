"""Plan 12 folds with rotating attack hosts and benign source-host groups."""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed/nfv3_directional")
    parser.add_argument("--output", default="reports/nfv3_full_host_folds.json")
    args = parser.parse_args()
    root = Path(args.data_root)
    frames = []
    for name in ("train", "calibration", "test"):
        path = root / f"{name}.csv"
        frame = pd.read_csv(path, usecols=["source_host_id", "source_host_group", "label_binary"], low_memory=False)
        frame["partition"] = name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    host_stats = []
    for host, group in frame.groupby("source_host_id"):
        host_stats.append({"host_id": str(host), "rows": int(len(group)), "attacks": int(group["label_binary"].sum())})
    attack_hosts = sorted(item["host_id"] for item in host_stats if item["attacks"] > 0)
    benign_groups = sorted(group for group in frame["source_host_group"].astype(str).unique() if group.startswith("benign_"))
    if len(attack_hosts) < 4 or len(benign_groups) < 3:
        raise ValueError(f"Need at least 4 attack hosts and 3 benign groups; found {len(attack_hosts)} and {len(benign_groups)}")
    attack_pairs = list(permutations(attack_hosts, 2))
    folds = []
    for index, (calibration_attack, test_attack) in enumerate(attack_pairs[:12]):
        train_attack = [host for host in attack_hosts if host not in {calibration_attack, test_attack}]
        offset = index % len(benign_groups)
        calibration_benign = benign_groups[offset]
        test_benign = benign_groups[(offset + 1) % len(benign_groups)]
        train_benign = [group for group in benign_groups if group not in {calibration_benign, test_benign}]
        folds.append({
            "train_attack_hosts": train_attack,
            "calibration_attack_host": calibration_attack,
            "test_attack_host": test_attack,
            "train_benign_groups": train_benign,
            "calibration_benign_groups": [calibration_benign],
            "test_benign_groups": [test_benign],
        })
    result = {
        "host_count": len(host_stats),
        "attack_host_count": len(attack_hosts),
        "benign_group_count": len(benign_groups),
        "benign_groups": benign_groups,
        "fold_count": len(folds),
        "folds": folds,
        "note": "Each fold holds out both attack-bearing hosts and benign source-host groups; thresholds must be selected on calibration only.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "attack_hosts": len(attack_hosts), "benign_groups": len(benign_groups), "folds": len(folds)}, indent=2))


if __name__ == "__main__":
    main()
