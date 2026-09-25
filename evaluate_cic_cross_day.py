"""Evaluate the saved Wednesday CIC-IDS2017 model on a different day.

No model retraining occurs. This checks cross-day transfer, not cross-dataset
or real-world generalization. It assumes both files use compatible CICFlowMeter
feature definitions.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from directionlab.dataset_adapters import prepare_labeled_csv


DEFAULT_MODEL = "reports/cic_ids2017_baseline/model.joblib"
DEFAULT_INPUT = (
    "data/raw/CIC-IDS2017/MachineLearningCVE/"
    "Tuesday-WorkingHours.pcap_ISCX.csv"
)
DEFAULT_OUTPUT = "reports/cic_ids2017_cross_day_tuesday"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    expected_features = list(bundle["feature_columns"])

    print(f"Loading target day: {input_path}")
    X, y, adapter_metadata = prepare_labeled_csv(input_path, task="binary")

    missing = [c for c in expected_features if c not in X.columns]
    if missing:
        raise ValueError(
            "Target dataset is missing model-required features: "
            + ", ".join(missing)
        )

    extra = [c for c in X.columns if c not in expected_features]
    X_eval = X.loc[:, expected_features]

    print(
        f"Evaluating saved model on {len(X_eval):,} rows "
        f"with {len(expected_features)} expected features..."
    )
    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_attack": float(precision_score(y, predictions, zero_division=0)),
        "recall_attack": float(recall_score(y, predictions, zero_division=0)),
        "f1_attack": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "input_path": str(input_path),
        "dataset_detected": adapter_metadata["detected_dataset"],
        "evaluation_design": (
            "Apply the saved model trained on the Wednesday random training "
            "split to a separate CIC-IDS2017 day. No refitting or threshold "
            "tuning on the target day."
        ),
        "rows_evaluated": int(len(X_eval)),
        "feature_count": len(expected_features),
        "extra_target_features_ignored": extra,
        "metrics": metrics,
        "confusion_matrix": {
            "labels": ["BENIGN", "ATTACK"],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            output_dict=True,
            zero_division=0,
        ),
        "limitations": [
            "This is cross-day evaluation within CIC-IDS2017, not cross-dataset validation.",
            "Feature names and definitions must be compatible across the two days.",
            "Dataset-specific collection artifacts may affect results.",
            "The saved model was trained on a random subset of Wednesday; the other 20% of Wednesday was not used here.",
        ],
    }

    report_path = output_dir / "evaluation.json"
    predictions_path = output_dir / "predictions.csv"

    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    output = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": predictions,
            "attack_score": probabilities,
        }
    )
    output.to_csv(predictions_path, index=False)

    print("\n=== CIC-IDS2017 cross-day results ===")
    print(f"Rows evaluated:    {len(X_eval):,}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Attack precision:  {metrics['precision_attack']:.4f}")
    print(f"Attack recall:     {metrics['recall_attack']:.4f}")
    print(f"Attack F1:         {metrics['f1_attack']:.4f}")
    print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.4%}")
    print(f"Confusion matrix:  TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print("\nSaved:")
    print(f"  Evaluation:  {report_path}")
    print(f"  Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
