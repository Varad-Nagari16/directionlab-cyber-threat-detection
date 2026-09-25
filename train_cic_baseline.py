"""Train and evaluate a reproducible CIC-IDS2017 binary baseline.

This is a stratified random split of one CSV/day. It is a pipeline smoke test,
not evidence of temporal, host-level, or cross-dataset generalization.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from directionlab.dataset_adapters import prepare_labeled_csv


DEFAULT_INPUT = (
    "data/raw/CIC-IDS2017/MachineLearningCVE/"
    "Wednesday-workingHours.pcap_ISCX.csv"
)
DEFAULT_OUTPUT = "reports/cic_ids2017_baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.50:
        parser.error("--test-size must be between 0.05 and 0.50")
    if args.trees < 1:
        parser.error("--trees must be at least 1")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    X, y, adapter_metadata = prepare_labeled_csv(
        input_path,
        task="binary",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.trees,
                    max_depth=args.max_depth,
                    class_weight="balanced_subsample",
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(
        f"Training on {len(X_train):,} rows; "
        f"evaluating on {len(X_test):,} rows..."
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Binary confusion matrix with fixed label order:
    # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "dataset_detected": adapter_metadata["detected_dataset"],
        "task": "binary",
        "evaluation_design": (
            "Stratified random train/test split within one CIC-IDS2017 "
            "Wednesday CSV. Not a temporal, host-independent, or "
            "cross-dataset evaluation."
        ),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "class_names": ["BENIGN", "ATTACK"],
        "train_class_counts": {
            "benign": int(np.sum(y_train == 0)),
            "attack": int(np.sum(y_train == 1)),
        },
        "test_class_counts": {
            "benign": int(np.sum(y_test == 0)),
            "attack": int(np.sum(y_test == 1)),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_attack": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall_attack": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_attack": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        },
        "confusion_matrix": {
            "labels": ["BENIGN", "ATTACK"],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            output_dict=True,
            zero_division=0,
        ),
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": args.trees,
            "max_depth": args.max_depth,
            "class_weight": "balanced_subsample",
            "imputation": "median; fit on training split only",
        },
        "adapter_metadata": adapter_metadata,
        "limitations": [
            "Random split may place related flows in both train and test.",
            "Metrics are not a cross-day or cross-dataset generalization result.",
            "False-positive rate is measured on this held-out split only.",
            "No threshold tuning was performed on the test set.",
        ],
    }

    model_path = output_dir / "model.joblib"
    report_path = output_dir / "evaluation.json"
    predictions_path = output_dir / "test_predictions.csv"

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "class_names": ["BENIGN", "ATTACK"],
            "random_state": args.random_state,
        },
        model_path,
    )

    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    prediction_frame = X_test.copy()
    prediction_frame["y_true"] = y_test
    prediction_frame["y_pred"] = predictions
    prediction_frame["attack_score"] = probabilities
    prediction_frame.to_csv(predictions_path, index=False)

    print("\n=== CIC-IDS2017 baseline results ===")
    print(f"Accuracy:           {report['metrics']['accuracy']:.4f}")
    print(f"Attack precision:   {report['metrics']['precision_attack']:.4f}")
    print(f"Attack recall:      {report['metrics']['recall_attack']:.4f}")
    print(f"Attack F1:          {report['metrics']['f1_attack']:.4f}")
    print(f"ROC AUC:            {report['metrics']['roc_auc']:.4f}")
    print(f"False-positive rate:{report['metrics']['false_positive_rate']:.4%}")
    print(f"Confusion matrix:   TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print("\nSaved:")
    print(f"  Model:       {model_path}")
    print(f"  Evaluation:  {report_path}")
    print(f"  Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
