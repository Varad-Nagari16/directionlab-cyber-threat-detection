"""Evaluate JSONL alerts against reference labels from an offline dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts", required=True)
    parser.add_argument("--output", default="reports/alert_evaluation.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.alerts).read_text().splitlines() if line.strip()]
    labeled = [row for row in rows if "reference_label" in row]
    if not labeled:
        raise ValueError("No reference_label fields found; this script needs offline labeled alerts")

    benign_names = {"BENIGN", "benign", "Normal", "normal"}
    y_true = [str(row["reference_label"]) not in benign_names for row in labeled]
    y_pred = [row.get("threat_class") == "attack" for row in labeled]
    tp = sum(a and b for a, b in zip(y_true, y_pred))
    tn = sum(not a and not b for a, b in zip(y_true, y_pred))
    fp = sum(not a and b for a, b in zip(y_true, y_pred))
    fn = sum(a and not b for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    families = defaultdict(lambda: Counter())
    for truth, pred, row in zip(y_true, y_pred, labeled):
        family = str(row["reference_label"])
        families[family]["rows"] += 1
        families[family]["predicted_attack"] += int(pred)
        families[family]["correct_attack"] += int(truth and pred)
        families[family]["missed_attack"] += int(truth and not pred)
        families[family]["false_alert"] += int(not truth and pred)

    result = {
        "rows_evaluated": len(labeled),
        "confusion_matrix": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": sum(y_pred) / len(y_pred),
        "actual_attack_rate": sum(y_true) / len(y_true),
        "families": {key: dict(value) for key, value in sorted(families.items())},
        "note": "Metrics apply only to the rows written to the JSONL file; reference labels are unavailable in real deployment.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
