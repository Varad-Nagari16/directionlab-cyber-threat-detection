"""Tune a Bot-specific rejection threshold on calibration alerts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if "reference_label" in row and "class_scores" in row]


def truth(label: str) -> bool:
    return str(label).strip().lower() == "bot"


def classify(row: dict, threshold: float) -> str:
    scores = row["class_scores"]
    predicted = max(scores, key=scores.get)
    if predicted == "Bot" and float(scores["Bot"]) < threshold:
        return max((name for name in scores if name != "Bot"), key=lambda name: scores[name])
    return predicted


def metrics(rows: list[dict], threshold: float) -> dict[str, float]:
    actual = np.array([truth(row["reference_label"]) for row in rows])
    predicted = np.array([classify(row, threshold) == "Bot" for row in rows])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "predicted_bot": int(predicted.sum()),
        "actual_bot": int(actual.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="reports/cicids_multiclass_family/bot_threshold.json")
    args = parser.parse_args()
    calibration, test = load(Path(args.calibration)), load(Path(args.test))
    values = np.array([float(row["class_scores"]["Bot"]) for row in calibration])
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 501)))
    best = max((metrics(calibration, float(value)) for value in candidates), key=lambda item: item["f1"])
    result = {"selected_on": "calibration", "calibration": best, "test": metrics(test, best["threshold"]), "warning": "The test result is for reporting only; do not retune using test data."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
