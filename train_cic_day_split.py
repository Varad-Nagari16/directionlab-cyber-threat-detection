from pathlib import Path
import gc
import json

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
DATA = ROOT / "data/raw/CIC-IDS2017/MachineLearningCVE"
OUT = ROOT / "reports/cic_day_split"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_PER_TRAIN_DAY = 100_000
CHUNK_SIZE = 50_000
TARGET_FPR = 0.01

TRAIN_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

VALIDATION_FILES = [
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

TEST_FILES = [
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]


def clean_columns(df):
    df.columns = [str(c).strip() for c in df.columns]

    # CIC files may contain a duplicate Fwd Header Length column.
    duplicate = "Fwd Header Length.1"
    original = "Fwd Header Length"

    if duplicate in df.columns and original in df.columns:
        if df[duplicate].equals(df[original]):
            df = df.drop(columns=[duplicate])

    return df


def find_label_column(columns):
    for col in columns:
        if str(col).strip().lower() == "label":
            return col
    raise ValueError("Could not find a Label column.")


def load_training_sample(filename, sample_size, seed):
    path = DATA / filename
    print(f"\nSampling training data: {filename}")

    # Read one training file at a time to limit peak RAM.
    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)

    label_col = find_label_column(df.columns)
    labels = df[label_col].astype(str).str.strip()

    # Binary task: BENIGN vs any labeled attack.
    y = (labels.str.upper() != "BENIGN").astype("int8")

    X = df.drop(columns=[label_col])

    # Keep numeric flow features only. Exclude identifiers and timestamps.
    drop_names = {
        "flow id", "source ip", "destination ip",
        "timestamp", "src ip", "dst ip",
    }
    X = X[
        [
            c for c in X.columns
            if c.strip().lower() not in drop_names
        ]
    ]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that have no usable numeric values in this file.
    X = X.dropna(axis=1, how="all")

    # Sample without replacement.
    n = min(sample_size, len(df))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(df), size=n, replace=False))

    X = X.iloc[indices].copy()
    y = y.iloc[indices].copy()

    print(
        f"  Sampled {len(X):,} rows; "
        f"attacks={int(y.sum()):,}; features={X.shape[1]}"
    )

    del df
    gc.collect()
    return X, y


def align_features(X, feature_names):
    # Reindex ensures validation/test use the training feature order.
    X = X.reindex(columns=feature_names)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def score_files(model, filenames, feature_names, description):
    scores = []
    labels_all = []
    attack_types = []

    for filename in filenames:
        path = DATA / filename
        print(f"\nScoring {description}: {filename}")

        for chunk in pd.read_csv(
            path, chunksize=CHUNK_SIZE, low_memory=False
        ):
            chunk = clean_columns(chunk)
            label_col = find_label_column(chunk.columns)

            labels = chunk[label_col].astype(str).str.strip()
            y = (labels.str.upper() != "BENIGN").astype("int8")

            X = chunk.drop(columns=[label_col])
            drop_names = {
                "flow id", "source ip", "destination ip",
                "timestamp", "src ip", "dst ip",
            }
            X = X[
                [
                    c for c in X.columns
                    if c.strip().lower() not in drop_names
                ]
            ]

            X = align_features(X, feature_names)

            score = model.predict_proba(X)[:, 1]

            scores.append(score)
            labels_all.append(y.to_numpy())
            attack_types.extend(labels.tolist())

    return (
        np.concatenate(labels_all),
        np.concatenate(scores),
        np.asarray(attack_types, dtype=object),
    )


def choose_threshold(y, scores, max_fpr):
    # Select the threshold with the highest recall while meeting
    # the FPR constraint on validation data.
    candidates = np.unique(scores)
    candidates = np.append(
        candidates,
        np.nextafter(scores.max(), np.inf),
    )

    best = None

    for threshold in candidates:
        pred = (scores >= threshold).astype("int8")

        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if fp + tn else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        if fpr <= max_fpr:
            candidate = (recall, -fpr, threshold)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        raise RuntimeError("Could not find a valid threshold.")

    return float(best[2])


def evaluate(y, scores, threshold, attack_types):
    pred = (scores >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    result = {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "attack_precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "attack_recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "attack_f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, scores))
    else:
        result["roc_auc"] = None

    breakdown = []
    for label in sorted(set(attack_types)):
        mask = attack_types == label
        actual = y[mask]
        predicted = pred[mask]

        breakdown.append({
            "attack_type": str(label),
            "flows": int(mask.sum()),
            "actual_attacks": int(actual.sum()),
            "detected": int(
                ((actual == 1) & (predicted == 1)).sum()
            ),
            "missed": int(
                ((actual == 1) & (predicted == 0)).sum()
            ),
            "recall": (
                float(
                    ((actual == 1) & (predicted == 1)).sum()
                    / actual.sum()
                )
                if actual.sum() else None
            ),
            "false_alarms": int(
                ((actual == 0) & (predicted == 1)).sum()
            ),
        })

    return result, breakdown


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main():
    # Load and sample training days separately.
    X_parts = []
    y_parts = []

    for i, filename in enumerate(TRAIN_FILES):
        X, y = load_training_sample(
            filename,
            SAMPLE_PER_TRAIN_DAY,
            SEED + i,
        )
        X_parts.append(X)
        y_parts.append(y)

    # Use only features available in both training days.
    common_features = list(
        set(X_parts[0].columns) & set(X_parts[1].columns)
    )
    common_features.sort()

    X_train = pd.concat(
        [x.reindex(columns=common_features) for x in X_parts],
        ignore_index=True,
    )
    y_train = pd.concat(y_parts, ignore_index=True)

    del X_parts, y_parts
    gc.collect()

    print("\nTraining rows:", f"{len(X_train):,}")
    print("Training features:", len(common_features))
    print("Training attacks:", f"{int(y_train.sum()):,}")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
            min_samples_leaf=2,
        )),
    ])

    print("\nFitting Random Forest...")
    model.fit(X_train, y_train)

    import joblib
    joblib.dump(model, OUT / "model.joblib")

    del X_train, y_train
    gc.collect()

    # Validation: select threshold using Thursday only.
    y_val, score_val, labels_val = score_files(
        model, VALIDATION_FILES, common_features, "validation"
    )

    threshold = choose_threshold(
        y_val, score_val, TARGET_FPR
    )

    val_metrics, val_breakdown = evaluate(
        y_val, score_val, threshold, labels_val
    )

    save_json(OUT / "validation.json", {
        "threshold": threshold,
        "target_fpr": TARGET_FPR,
        "metrics": val_metrics,
        "attack_breakdown": val_breakdown,
    })

    print("\nSelected threshold:", threshold)
    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))

    # Final test: Friday is not used for fitting or threshold choice.
    y_test, score_test, labels_test = score_files(
        model, TEST_FILES, common_features, "final test"
    )

    test_metrics, test_breakdown = evaluate(
        y_test, score_test, threshold, labels_test
    )

    save_json(OUT / "friday_test.json", {
        "threshold_from_thursday": threshold,
        "metrics": test_metrics,
        "attack_breakdown": test_breakdown,
    })

    print("\nFINAL FRIDAY TEST:")
    print(json.dumps(test_metrics, indent=2))
    print("\nPer-label breakdown:")
    print(pd.DataFrame(test_breakdown).to_string(index=False))

    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()