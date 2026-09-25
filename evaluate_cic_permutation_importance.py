from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "CIC-IDS2017" / "MachineLearningCVE"
OUT_DIR = ROOT / "reports" / "cic_all_attack_coverage"

PREDICTIONS_PATH = OUT_DIR / "test_predictions.csv"
MODEL_PATH = OUT_DIR / "model.joblib"
IMPURITY_PATH = OUT_DIR / "feature_importance.csv"

OUTPUT_PATH = OUT_DIR / "permutation_importance_by_file.csv"

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

TOP_N = 25
MAX_PER_CLASS = 2500
N_REPEATS = 3


def read_cic_csv(path):
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="cp1252",
            low_memory=False,
        )

    df.columns = [str(c).strip() for c in df.columns]

    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"),
        None,
    )

    if label_col is None:
        raise ValueError(f"No Label column in {path.name}")

    # Remove duplicate semantic columns, e.g. Fwd Header Length.1
    drop_cols = [
        col for col in df.columns
        if col.endswith(".1") and col[:-2] in df.columns
    ]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    labels = (
        df[label_col]
        .astype("string")
        .str.strip()
        .fillna("UNKNOWN")
    )

    features = df.drop(columns=[label_col])

    for col in features.columns:
        features[col] = pd.to_numeric(
            features[col],
            errors="coerce",
        )

    features = features.dropna(axis=1, how="all")
    features = features.replace([np.inf, -np.inf], np.nan)

    return features, labels


def reconstruct_test_indices(labels, seed):
    """
    Reproduce the original split exactly.

    Returns test row indices in the same sorted-label order
    used by the training script.
    """
    rng = np.random.default_rng(seed)
    test_parts = []

    for label in sorted(labels.unique()):
        indices = np.flatnonzero(
            labels.to_numpy() == label
        )
        rng.shuffle(indices)

        n = len(indices)

        if n < 3:
            raise ValueError(
                f"Label {label!r} has only {n} rows."
            )

        n_test = max(1, int(round(n * TEST_SIZE)))
        n_val = max(1, int(round(n * VALIDATION_SIZE)))
        n_train = n - n_test - n_val

        while n_train < 1:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError(
                    f"Cannot split label {label!r}"
                )

            n_train = n - n_test - n_val

        test_idx = indices[n_train + n_val:]
        test_parts.append(test_idx)

    return np.concatenate(test_parts)


def main():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    saved = pd.read_csv(PREDICTIONS_PATH)
    model = load(MODEL_PATH)
    impurity = pd.read_csv(IMPURITY_PATH)

    feature_columns = None

    X_parts = []
    label_parts = []
    file_parts = []

    # Reconstruct the exact test rows from the original captures.
    for file_idx, path in enumerate(files):
        print(f"Reconstructing {path.name}...")

        X, labels = read_cic_csv(path)

        if feature_columns is None:
            feature_columns = list(X.columns)
        else:
            X = X.reindex(columns=feature_columns)

        test_idx = reconstruct_test_indices(
            labels,
            SEED + file_idx,
        )

        X_test_file = X.iloc[test_idx].reset_index(drop=True)
        labels_test_file = (
            labels.iloc[test_idx].reset_index(drop=True)
        )

        X_parts.append(X_test_file)
        label_parts.append(labels_test_file)

        file_parts.append(
            pd.Series([path.name] * len(test_idx))
        )

    X_test = pd.concat(X_parts, ignore_index=True)
    labels_test = pd.concat(label_parts, ignore_index=True)
    files_test = pd.concat(file_parts, ignore_index=True)

    # Verify that reconstructed rows match saved predictions.
    if len(saved) != len(X_test):
        raise RuntimeError(
            f"Row-count mismatch: predictions={len(saved):,}, "
            f"reconstructed={len(X_test):,}"
        )

    if not np.array_equal(
        saved["source_file"].astype(str).to_numpy(),
        files_test.astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "Source-file order mismatch. Stop: predictions "
            "are not aligned."
        )

    if not np.array_equal(
        saved["true_label"].astype(str).to_numpy(),
        labels_test.astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "Label order mismatch. Stop: predictions "
            "are not aligned."
        )

    # Ensure the reconstructed schema matches the fitted model.
    model_features = list(model.feature_names_in_)

    if list(X_test.columns) != model_features:
        raise RuntimeError(
            "Feature schema/order differs from the fitted model."
        )

    print(f"\nAlignment verified: {len(saved):,} rows match.")
    print(f"Feature count: {X_test.shape[1]}")

    # Select the top features from the existing impurity report.
    top_features = (
        impurity.sort_values(
            "importance",
            ascending=False,
        )["feature"]
        .head(TOP_N)
        .tolist()
    )

    missing = [
        feature for feature in top_features
        if feature not in X_test.columns
    ]

    if missing:
        raise RuntimeError(
            f"Features missing from reconstructed data: {missing}"
        )

    y_all = saved["y_true"].to_numpy()
    file_names = files_test.to_numpy()

    rng = np.random.default_rng(SEED)
    results = []

    # Analyze each capture separately.
    for file_name in pd.unique(file_names):
        file_positions = np.flatnonzero(
            file_names == file_name
        )

        y_file = y_all[file_positions]

        benign_positions = file_positions[y_file == 0]
        attack_positions = file_positions[y_file == 1]

        print(
            f"\n{file_name}\n"
            f"  Test rows: {len(file_positions):,}\n"
            f"  Benign: {len(benign_positions):,}\n"
            f"  Attack: {len(attack_positions):,}"
        )

        # ROC AUC is undefined if only one class is present.
        if (
            len(benign_positions) == 0
            or len(attack_positions) == 0
        ):
            print("  Skipping: only one class in this capture.")
            continue

        # Sample each class separately. This makes rare attacks
        # less likely to disappear from the evaluation sample.
        if len(benign_positions) > MAX_PER_CLASS:
            benign_positions = rng.choice(
                benign_positions,
                size=MAX_PER_CLASS,
                replace=False,
            )

        if len(attack_positions) > MAX_PER_CLASS:
            attack_positions = rng.choice(
                attack_positions,
                size=MAX_PER_CLASS,
                replace=False,
            )

        sample_positions = np.concatenate([
            benign_positions,
            attack_positions,
        ])

        rng.shuffle(sample_positions)

        # IMPORTANT: pass all model features, not just the top 25.
        X_sample = X_test.iloc[sample_positions].copy()
        y_sample = y_all[sample_positions]

        baseline_scores = model.predict_proba(
            X_sample
        )[:, 1]

        baseline_auc = roc_auc_score(
            y_sample,
            baseline_scores,
        )

        print(
            f"  Sample rows: {len(sample_positions):,}\n"
            f"  Baseline AUC: {baseline_auc:.6f}"
        )

        for feature in top_features:
            drops = []

            for repeat in range(N_REPEATS):
                X_permuted = X_sample.copy()

                values = X_permuted[feature].to_numpy(
                    copy=True
                )

                X_permuted[feature] = rng.permutation(values)

                permuted_scores = model.predict_proba(
                    X_permuted
                )[:, 1]

                permuted_auc = roc_auc_score(
                    y_sample,
                    permuted_scores,
                )

                drops.append(
                    baseline_auc - permuted_auc
                )

            results.append({
                "source_file": file_name,
                "rows_in_capture_test_split": int(
                    len(file_positions)
                ),
                "sample_rows": int(len(sample_positions)),
                "benign_sampled": int(
                    np.sum(y_sample == 0)
                ),
                "attack_sampled": int(
                    np.sum(y_sample == 1)
                ),
                "baseline_auc": float(baseline_auc),
                "feature": feature,
                "permutation_auc_drop_mean": float(
                    np.mean(drops)
                ),
                "permutation_auc_drop_std": float(
                    np.std(drops)
                ),
            })

    if not results:
        raise RuntimeError(
            "No capture had both benign and attack rows."
        )

    output = pd.DataFrame(results)

    output = output.sort_values(
        ["source_file", "permutation_auc_drop_mean"],
        ascending=[True, False],
    )

    output.to_csv(OUTPUT_PATH, index=False)

    print("\n=== TOP FEATURES BY CAPTURE ===")

    for file_name, group in output.groupby(
        "source_file",
        sort=False,
    ):
        print(f"\n{file_name}")

        print(
            group.head(10)[
                [
                    "feature",
                    "permutation_auc_drop_mean",
                    "permutation_auc_drop_std",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()