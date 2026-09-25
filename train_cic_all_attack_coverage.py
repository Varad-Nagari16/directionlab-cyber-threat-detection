from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "CIC-IDS2017" / "MachineLearningCVE"
OUT_DIR = ROOT / "reports" / "cic_all_attack_coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_CAP_PER_LABEL_PER_FILE = 50_000
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
TARGET_FPR = 0.01

BENIGN_LABEL = "BENIGN"


def read_cic_csv(path):
    """Read a CIC CSV and remove duplicate semantic columns."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252", low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]

    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"),
        None,
    )
    if label_col is None:
        raise ValueError(f"No Label column in {path.name}")

    # CICFlowMeter data can contain a duplicate "Fwd Header Length".
    # pandas typically renames the duplicate to ".1".
    drop_cols = []
    for col in df.columns:
        if col.endswith(".1") and col[:-2] in df.columns:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    labels = (
        df[label_col]
        .astype("string")
        .str.strip()
        .fillna("UNKNOWN")
    )

    feature_df = df.drop(columns=[label_col])

    # Keep numeric features; coerce malformed numeric values to NaN.
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(
            feature_df[col], errors="coerce"
        )

    # Remove columns that contain no usable numeric values.
    feature_df = feature_df.dropna(axis=1, how="all")

    # Replace infinities before imputation.
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    return feature_df, labels


def split_one_file(features, labels, file_name, seed):
    """
    Stratified-by-label split within a file.

    For each label, shuffle its rows and assign approximately:
    60% train, 20% validation, 20% test.
    """
    rng = np.random.default_rng(seed)

    train_parts = []
    val_parts = []
    test_parts = []

    for label in sorted(labels.unique()):
        indices = np.flatnonzero(labels.to_numpy() == label)
        rng.shuffle(indices)

        n = len(indices)
        if n < 3:
            raise ValueError(
                f"{file_name}: label {label!r} has only {n} rows; "
                "cannot put it in all three splits."
            )

        n_test = max(1, int(round(n * TEST_SIZE)))
        n_val = max(1, int(round(n * VALIDATION_SIZE)))
        n_train = n - n_test - n_val

        # Ensure at least one training row for this label.
        while n_train < 1:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError(
                    f"Cannot split label {label!r} in {file_name}"
                )
            n_train = n - n_test - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_parts.append((features.iloc[train_idx], labels.iloc[train_idx]))
        val_parts.append((features.iloc[val_idx], labels.iloc[val_idx]))
        test_parts.append((features.iloc[test_idx], labels.iloc[test_idx]))

    def combine(parts):
        x = pd.concat([p[0] for p in parts], ignore_index=True)
        y = pd.concat([p[1] for p in parts], ignore_index=True)
        return x, y

    x_train, y_train = combine(train_parts)
    x_val, y_val = combine(val_parts)
    x_test, y_test = combine(test_parts)

    return (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
    )


def cap_training_rows(x, y, cap, seed):
    """Cap each label's training rows; leave validation/test untouched."""
    rng = np.random.default_rng(seed)
    keep = []

    for label in sorted(y.unique()):
        idx = np.flatnonzero(y.to_numpy() == label)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.extend(idx.tolist())

    keep = np.asarray(keep, dtype=int)
    rng.shuffle(keep)

    return x.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def align_features(frame, columns):
    return frame.reindex(columns=columns)


def binary_target(labels):
    return (labels.astype(str) != BENIGN_LABEL).astype(np.uint8)


def choose_threshold_for_fpr(y_true, scores, target_fpr):
    """Choose a validation threshold whose observed benign FPR is <= target."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    benign_scores = scores[y_true == 0]
    if len(benign_scores) == 0:
        raise ValueError("Validation set has no benign rows.")

    candidates = np.unique(benign_scores)
    candidates = np.sort(candidates)

    # Include a threshold above the maximum score (zero false positives).
    candidates = np.append(candidates, np.nextafter(candidates[-1], np.inf))

    valid = []
    for threshold in candidates:
        fpr = float(np.mean(benign_scores >= threshold))
        if fpr <= target_fpr:
            valid.append((threshold, fpr))

    # Lowest valid threshold maximizes recall among thresholds meeting target.
    return float(valid[0][0])


def evaluate(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "attack_precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "attack_recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "attack_f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores))
        if len(np.unique(y_true)) == 2 else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }, pred


def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    train_x_parts, train_y_parts = [], []
    val_x_parts, val_y_parts = [], []
    test_x_parts, test_y_parts = [], []
    test_label_parts, test_file_parts = [], []

    feature_columns = None

    for file_idx, path in enumerate(files):
        print(f"\nLoading {path.name}")
        x, labels = read_cic_csv(path)

        # Establish a shared feature schema, excluding duplicate columns.
        if feature_columns is None:
            feature_columns = list(x.columns)
        else:
            missing = set(feature_columns) - set(x.columns)
            extra = set(x.columns) - set(feature_columns)
            if missing or extra:
                print(
                    f"  Feature schema differs: "
                    f"{len(missing)} missing, {len(extra)} extra"
                )
            x = x.reindex(columns=feature_columns)

        print(f"  Rows: {len(x):,}; labels: {labels.nunique()}")

        (xtr, ytr), (xv, yv), (xt, yt) = split_one_file(
            x, labels, path.name, SEED + file_idx
        )

        xtr, ytr = cap_training_rows(
            xtr,
            ytr,
            TRAIN_CAP_PER_LABEL_PER_FILE,
            SEED + 100 + file_idx,
        )

        print(
            f"  Train: {len(xtr):,}; "
            f"validation: {len(xv):,}; test: {len(xt):,}"
        )

        train_x_parts.append(xtr)
        train_y_parts.append(ytr)
        val_x_parts.append(xv)
        val_y_parts.append(yv)
        test_x_parts.append(xt)
        test_y_parts.append(binary_target(yt))
        test_label_parts.append(yt.reset_index(drop=True))
        test_file_parts.append(
            pd.Series([path.name] * len(yt), name="source_file")
        )

        del x, labels, xtr, ytr, xv, yv, xt, yt

    X_train = pd.concat(train_x_parts, ignore_index=True)
    y_train_labels = pd.concat(train_y_parts, ignore_index=True)
    y_train = binary_target(y_train_labels)

    X_val = pd.concat(val_x_parts, ignore_index=True)
    y_val_labels = pd.concat(val_y_parts, ignore_index=True)
    y_val = binary_target(y_val_labels)

    X_test = pd.concat(test_x_parts, ignore_index=True)
    y_test = pd.concat(test_y_parts, ignore_index=True)
    test_labels = pd.concat(test_label_parts, ignore_index=True)
    test_files = pd.concat(test_file_parts, ignore_index=True)

    print("\nTraining Random Forest...")
    print(f"Training rows: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        )),
    ])

    model.fit(X_train, y_train)

    print("Scoring validation set...")
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold_for_fpr(
        y_val, val_scores, TARGET_FPR
    )
    val_metrics, _ = evaluate(y_val, val_scores, threshold)

    print("Scoring held-out test set...")
    test_scores = model.predict_proba(X_test)[:, 1]
    overall, test_pred = evaluate(y_test, test_scores, threshold)

    # Per-source-file and per-original-label metrics.
    breakdown = []
    for label in sorted(test_labels.unique()):
        mask = (test_labels.to_numpy() == label)
        if label == BENIGN_LABEL:
            # For benign rows, report false-positive rate directly.
            fp = int(test_pred[mask].sum())
            breakdown.append({
                "label": label,
                "rows": int(mask.sum()),
                "detected_as_attack": fp,
                "missed_as_attack": int(mask.sum() - fp),
                "recall_or_fpr": float(fp / mask.sum()),
            })
        else:
            detected = int(test_pred[mask].sum())
            breakdown.append({
                "label": label,
                "rows": int(mask.sum()),
                "detected_as_attack": detected,
                "missed_as_attack": int(mask.sum() - detected),
                "recall_or_fpr": float(detected / mask.sum()),
            })

    per_file = []
    for file_name in sorted(test_files.unique()):
        mask = (test_files.to_numpy() == file_name)
        metrics, _ = evaluate(y_test.to_numpy()[mask],
                              test_scores[mask], threshold)
        per_file.append({"file": file_name, **metrics})

    model_path = OUT_DIR / "model.joblib"
    from joblib import dump
    dump(model, model_path)

    # Save predictions with source capture and original attack label.
    pd.DataFrame({
        "source_file": test_files,
        "true_label": test_labels,
        "y_true": y_test,
        "score": test_scores,
        "prediction": test_pred,
    }).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    pd.DataFrame(breakdown).to_csv(
        OUT_DIR / "per_attack_metrics.csv", index=False
    )
    pd.DataFrame(per_file).to_csv(
        OUT_DIR / "per_file_metrics.csv", index=False
    )

    report = {
        "experiment": "CIC-IDS2017 within-capture all-attack coverage",
        "warning": (
            "Random stratified rows within each capture. Related flows may "
            "cross splits; this is not independent-capture generalization."
        ),
        "seed": SEED,
        "training_cap_per_label_per_file": TRAIN_CAP_PER_LABEL_PER_FILE,
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "target_validation_fpr": TARGET_FPR,
        "selected_threshold": float(threshold),
        "validation_metrics_at_threshold": val_metrics,
        "test_metrics": overall,
        "model_path": str(model_path),
    }

    with open(OUT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== ALL-ATTACK COVERAGE RESULTS ===")
    print(json.dumps(report, indent=2))
    print(f"\nPer-attack metrics: {OUT_DIR / 'per_attack_metrics.csv'}")
    print(f"Per-file metrics:   {OUT_DIR / 'per_file_metrics.csv'}")
    print(f"Predictions:        {OUT_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()