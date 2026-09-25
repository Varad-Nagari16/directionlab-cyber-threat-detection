from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "CIC-IDS2017" / "MachineLearningCVE"
OUT_DIR = ROOT / "reports" / "cic_capture_heldout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_CAP_PER_LABEL_PER_FILE = 50_000
VALIDATION_FRACTION = 0.20
TARGET_FPR = 0.01
N_ESTIMATORS = 200
BENIGN_LABEL = "BENIGN"
DROP_DUPLICATE_SUFFIX_COLUMNS = True


def read_cic_csv(path):
    """Read a CIC CSV, clean labels, and retain numeric flow features."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252", low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        raise ValueError(f"No Label column in {path.name}")

    if DROP_DUPLICATE_SUFFIX_COLUMNS:
        drop_cols = [
            c for c in df.columns
            if c.endswith(".1") and c[:-2] in df.columns
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    labels = (
        df[label_col].astype("string").str.strip().fillna("UNKNOWN")
    )
    features = df.drop(columns=[label_col]).copy()

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    features = features.dropna(axis=1, how="all")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, labels


def binary_target(labels):
    return (labels.astype(str) != BENIGN_LABEL).astype(np.uint8)


def split_training_capture(features, labels, seed):
    """
    Split rows from a training capture into model-training and validation
    subsets, stratified by original label where possible. Tiny labels remain
    in training so that at least one row is available to fit.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    y = labels.to_numpy()

    for label in sorted(labels.unique()):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n = len(idx)

        # Keep at least one row for fitting. Avoid fragile validation samples
        # for labels with fewer than five rows.
        if n >= 5:
            n_val = max(1, int(round(n * VALIDATION_FRACTION)))
            n_val = min(n_val, n - 1)
        else:
            n_val = 0

        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    x_train = features.iloc[train_idx].reset_index(drop=True)
    y_train = labels.iloc[train_idx].reset_index(drop=True)
    x_val = features.iloc[val_idx].reset_index(drop=True)
    y_val = labels.iloc[val_idx].reset_index(drop=True)
    return x_train, y_train, x_val, y_val


def cap_training_rows(x, y, cap, seed):
    rng = np.random.default_rng(seed)
    keep = []
    y_values = y.to_numpy()

    for label in sorted(y.unique()):
        idx = np.flatnonzero(y_values == label)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.extend(idx.tolist())

    keep = np.asarray(keep, dtype=int)
    rng.shuffle(keep)
    return (
        x.iloc[keep].reset_index(drop=True),
        y.iloc[keep].reset_index(drop=True),
    )


def choose_threshold_for_fpr(y_true, scores, target_fpr):
    """Choose the lowest threshold with observed validation benign FPR <= target."""
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores)
    benign_scores = scores[y_true == 0]
    if len(benign_scores) == 0:
        raise ValueError("Validation set has no benign rows; cannot set FPR threshold.")

    candidates = np.sort(np.unique(benign_scores))
    candidates = np.append(candidates, np.nextafter(candidates[-1], np.inf))
    valid = [
        (threshold, float(np.mean(benign_scores >= threshold)))
        for threshold in candidates
        if float(np.mean(benign_scores >= threshold)) <= target_fpr
    ]
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
        "attack_precision": float(precision_score(y_true, pred, zero_division=0)),
        "attack_recall": float(recall_score(y_true, pred, zero_division=0)),
        "attack_f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, scores))
            if len(np.unique(y_true)) == 2 else None
        ),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }, pred


def load_capture(path):
    x, labels = read_cic_csv(path)
    return x, labels


def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    if len(files) < 3:
        raise FileNotFoundError(
            f"Expected at least 3 CIC CSV files in {DATA_DIR}; found {len(files)}"
        )

    print(f"Found {len(files)} captures.")
    print("Loading capture schemas...")
    # Derive a consistent feature schema across all captures. Using the
    # intersection avoids creating all-NaN columns for absent features.
    schema_by_file = {}
    for path in files:
        x, _ = load_capture(path)
        schema_by_file[path.name] = list(x.columns)
        print(f"  {path.name}: {len(x.columns)} features")

    first_columns = schema_by_file[files[0].name]
    common = set(first_columns)
    for path in files[1:]:
        common.intersection_update(schema_by_file[path.name])
    feature_columns = [c for c in first_columns if c in common]

    if not feature_columns:
        raise ValueError("No common numeric features found across capture files.")
    if "Destination Port" not in feature_columns:
        print("Note: Destination Port is not in the common feature schema.")

    variants = {
        "all_features": feature_columns,
        "without_destination_port": [
            c for c in feature_columns if c != "Destination Port"
        ],
    }
    if not variants["without_destination_port"]:
        raise ValueError("No features remain after excluding Destination Port.")

    all_results = []
    per_label_results = []

    for held_idx, held_path in enumerate(files):
        print(f"\n{'=' * 78}")
        print(f"HOLDOUT CAPTURE: {held_path.name}")
        print(f"{'=' * 78}")

        # Load the held-out capture once; it is never used for fitting or
        # threshold selection.
        x_test_full, labels_test = load_capture(held_path)
        x_test_full = x_test_full.reindex(columns=feature_columns)
        y_test = binary_target(labels_test).to_numpy()

        train_x_parts, train_y_parts, val_x_parts, val_y_parts = [], [], [], []

        for train_idx, train_path in enumerate(files):
            if train_path == held_path:
                continue

            x, labels = load_capture(train_path)
            x = x.reindex(columns=feature_columns)

            xtr, ytr, xv, yv = split_training_capture(
                x, labels, SEED + held_idx * 1000 + train_idx
            )
            xtr, ytr = cap_training_rows(
                xtr, ytr, TRAIN_CAP_PER_LABEL_PER_FILE,
                SEED + 100 + held_idx * 1000 + train_idx
            )

            train_x_parts.append(xtr)
            train_y_parts.append(ytr)
            if len(xv):
                val_x_parts.append(xv)
                val_y_parts.append(yv)

            print(
                f"  Train capture {train_path.name}: "
                f"fit={len(xtr):,}, validation={len(xv):,}"
            )
            del x, labels, xtr, ytr, xv, yv

        X_train_full = pd.concat(train_x_parts, ignore_index=True)
        y_train = binary_target(pd.concat(train_y_parts, ignore_index=True)).to_numpy()
        X_val_full = pd.concat(val_x_parts, ignore_index=True)
        y_val = binary_target(pd.concat(val_y_parts, ignore_index=True)).to_numpy()

        for variant_idx, (variant_name, cols) in enumerate(variants.items()):
            print(f"\n  Training variant: {variant_name} ({len(cols)} features)")
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=SEED + held_idx * 10 + variant_idx,
                )),
            ])

            model.fit(X_train_full[cols], y_train)
            val_scores = model.predict_proba(X_val_full[cols])[:, 1]
            threshold = choose_threshold_for_fpr(
                y_val, val_scores, TARGET_FPR
            )

            test_scores = model.predict_proba(x_test_full[cols])[:, 1]
            metrics, test_pred = evaluate(y_test, test_scores, threshold)
            result = {
                "heldout_capture": held_path.name,
                "variant": variant_name,
                "feature_count": len(cols),
                "train_captures": len(files) - 1,
                "train_rows": int(len(X_train_full)),
                "validation_rows": int(len(X_val_full)),
                "test_rows": int(len(x_test_full)),
                "target_validation_fpr": TARGET_FPR,
                **metrics,
            }
            all_results.append(result)
            print(
                f"    test recall={metrics['attack_recall']:.6f}, "
                f"FPR={metrics['fpr']}, AUC={metrics['roc_auc']}"
            )

            # Per-original-label results on the held-out capture.
            for label in sorted(labels_test.unique()):
                mask = labels_test.to_numpy() == label
                is_benign = label == BENIGN_LABEL
                if is_benign:
                    false_positives = int(test_pred[mask].sum())
                    per_label_results.append({
                        "heldout_capture": held_path.name,
                        "variant": variant_name,
                        "label": label,
                        "rows": int(mask.sum()),
                        "detected_as_attack": false_positives,
                        "missed_as_attack": int(mask.sum() - false_positives),
                        "recall_or_fpr": (
                            float(false_positives / mask.sum()) if mask.sum() else None
                        ),
                        "metric_type": "false_positive_rate",
                    })
                else:
                    detected = int(test_pred[mask].sum())
                    per_label_results.append({
                        "heldout_capture": held_path.name,
                        "variant": variant_name,
                        "label": label,
                        "rows": int(mask.sum()),
                        "detected_as_attack": detected,
                        "missed_as_attack": int(mask.sum() - detected),
                        "recall_or_fpr": (
                            float(detected / mask.sum()) if mask.sum() else None
                        ),
                        "metric_type": "attack_recall",
                    })

            del model, val_scores, test_scores, test_pred

        del x_test_full, labels_test, X_train_full, y_train, X_val_full, y_val

    results_df = pd.DataFrame(all_results)
    labels_df = pd.DataFrame(per_label_results)
    results_df.to_csv(OUT_DIR / "capture_heldout_summary.csv", index=False)
    labels_df.to_csv(OUT_DIR / "capture_heldout_per_label.csv", index=False)

    report = {
        "experiment": "CIC-IDS2017 leave-one-capture-out evaluation",
        "warning": (
            "Each test capture is excluded entirely from fitting and threshold "
            "selection. However, CIC-IDS2017 captures are from the same dataset "
            "and collection environment; this is not independent external validation. "
            "Attack labels absent from training are evaluated as binary attacks, "
            "not as known attack subclasses."
        ),
        "seed": SEED,
        "target_validation_fpr": TARGET_FPR,
        "training_cap_per_label_per_file": TRAIN_CAP_PER_LABEL_PER_FILE,
        "n_captures": len(files),
        "common_feature_count": len(feature_columns),
        "feature_count_without_destination_port": len(variants["without_destination_port"]),
        "results": all_results,
    }
    with open(OUT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:")
    print(f"  Summary:   {OUT_DIR / 'capture_heldout_summary.csv'}")
    print(f"  Per-label: {OUT_DIR / 'capture_heldout_per_label.csv'}")
    print(f"  Report:    {OUT_DIR / 'evaluation.json'}")
    print("\nNote: this runs two Random Forest models per held-out capture and may take a while.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
