"""Evaluate multiclass JSONL alerts against offline reference labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["BENIGN", "DDoS", "PortScan", "Bot", "OTHER_ATTACK"]


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
    parser.add_argument("--alerts", required=True)
    parser.add_argument("--output", default="reports/multiclass_alert_evaluation.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.alerts).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "reference_label" in row]
    truth = [family(row["reference_label"]) for row in rows]
    predicted = [str(row["threat_class"]) for row in rows]
    report = classification_report(truth, predicted, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)
    result = {
        "rows_evaluated": len(rows),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=CLASSES).tolist(),
        "per_class": {name: {metric: float(values[metric]) for metric in ("precision", "recall", "f1-score", "support")} for name, values in report.items() if name in CLASSES},
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "note": "This evaluates only rows written to the JSONL file; random family-split results are not temporal generalization results.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
