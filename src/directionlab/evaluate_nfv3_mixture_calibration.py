from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

from directionlab.models import DirectionalThreatModel


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def make_features(df, feature_columns):
    out = pd.DataFrame(index=df.index)

    for col in feature_columns:
        out[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0.0)

    return out.astype(np.float32)


def score_dataframe(df, checkpoint, device):
    feature_columns = checkpoint["feature_columns"]
    scaler_mean = np.asarray(
        checkpoint["scaler_mean"], dtype=np.float32
    )
    scaler_scale = np.asarray(
        checkpoint["scaler_scale"], dtype=np.float32
    )
    history = int(checkpoint["config"]["history"])
    hidden_dim = int(checkpoint["config"]["hidden_dim"])

    X = make_features(
        df, feature_columns
    ).to_numpy(dtype=np.float32)

    X = (
        X - scaler_mean
    ) / np.where(
        scaler_scale == 0,
        1.0,
        scaler_scale,
    )

    windows = np.zeros(
        (len(X), history, X.shape[1]),
        dtype=np.float32,
    )

    for end in range(len(X)):
        start = max(0, end - history + 1)
        chunk = X[start:end + 1]
        windows[end, -len(chunk):] = chunk

    model = DirectionalThreatModel(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        classes=2,
    )

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    scores = []
    batch_size = 2048

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))

            current = torch.from_numpy(
                X[start:end]
            ).to(device)

            hist = torch.from_numpy(
                windows[start:end]
            ).to(device)

            output = model(
                current,
                hist,
            )["binary_logit"]

            scores.append(
                torch.sigmoid(output)
                .cpu()
                .numpy()
            )

    return np.concatenate(scores)


def evaluate(df, scores, threshold):
    y = (~df["Label"].astype(str).str.upper().eq("BENIGN")).astype(int).to_numpy()
    pred = scores >= threshold

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    benign = max(tn + fp, 1)
    attacks = max(tp + fn, 1)

    start = pd.to_numeric(
        df["FLOW_START_MILLISECONDS"],
        errors="coerce",
    )
    end = pd.to_numeric(
        df["FLOW_END_MILLISECONDS"],
        errors="coerce",
    )

    hours = max(
        (end.max() - start.min()) / 1000.0 / 3600.0,
        1e-9,
    )

    return {
        "rows": int(len(df)),
        "benign_rows": int(benign),
        "attack_rows": int(attacks),
        "false_alerts": int(fp),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "fpr": float(fp / benign),
        "attack_recall": float(tp / attacks),
        "false_alerts_per_hour": float(fp / hours),
        "threshold": float(threshold),
        "time_span_hours": float(hours),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--adaptation-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--output-root", required=True)

    parser.add_argument(
        "--benign-quantile",
        type=float,
        default=0.99,
    )

    args = parser.parse_args()

    fold_root = Path(args.fold_root)
    adaptation_root = Path(args.adaptation_root)
    reports_root = Path(args.reports_root)
    output_root = Path(args.output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    rows = []

    for fold in range(1, 13):
        fold_name = f"fold_{fold:02d}"

        report_dir = reports_root / fold_name
        adapt_dir = adaptation_root / fold_name

        checkpoint_path = (
            report_dir / "model.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        adaptation = pd.read_csv(
            adapt_dir / "adaptation.csv"
        )

        evaluation = pd.read_csv(
            adapt_dir / "evaluation.csv"
        )

        adaptation_scores = score_dataframe(
            adaptation,
            checkpoint,
            device,
        )

        evaluation_scores = score_dataframe(
            evaluation,
            checkpoint,
            device,
        )

        # Calibration uses adaptation scores only.
        z = logit(
            adaptation_scores
        ).reshape(-1, 1)

        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=42,
            n_init=5,
        )

        gmm.fit(z)

        component_means = (
            gmm.means_.ravel()
        )

        # Lower-score component is the
        # inferred benign candidate.
        benign_component = int(
            np.argmin(component_means)
        )

        assignments = gmm.predict(z)

        inferred_benign_scores = (
            adaptation_scores[
                assignments
                == benign_component
            ]
        )

        if len(inferred_benign_scores) == 0:
            raise RuntimeError(
                f"{fold_name}: empty "
                "inferred benign component"
            )

        threshold = float(
            np.quantile(
                inferred_benign_scores,
                args.benign_quantile,
            )
        )

        result = evaluate(
            evaluation,
            evaluation_scores,
            threshold,
        )

        # Audit only. Labels are not used
        # during calibration.
        adaptation_attack_rate = float((~adaptation["Label"].astype(str).str.upper().eq("BENIGN")).astype(int).mean())

        inferred_benign_fraction = (
            len(inferred_benign_scores)
            / len(adaptation_scores)
        )

        result.update(
            {
                "fold": fold_name,
                "method": (
                    "two_component_logit_gmm_"
                    "empirical_benign_tail"
                ),
                "benign_quantile": (
                    args.benign_quantile
                ),
                "adaptation_rows": int(
                    len(adaptation)
                ),
                "adaptation_attack_rate_audit": (
                    adaptation_attack_rate
                ),
                "inferred_benign_component": (
                    benign_component
                ),
                "inferred_benign_rows": int(
                    len(inferred_benign_scores)
                ),
                "inferred_benign_fraction": (
                    float(inferred_benign_fraction)
                ),
                "component_mean_logit_0": float(
                    component_means[0]
                ),
                "component_mean_logit_1": float(
                    component_means[1]
                ),
                "device": str(device),
            }
        )

        rows.append(result)

        print(
            f"{fold_name}: "
            f"threshold={threshold:.6f}, "
            f"FPR={result['fpr']:.6f}, "
            f"recall={result['attack_recall']:.6f}, "
            f"alerts/hr="
            f"{result['false_alerts_per_hour']:.6f}, "
            f"inferred_benign="
            f"{inferred_benign_fraction:.3f}"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_root
        / "mixture_calibration_metrics.csv",
        index=False,
    )

    summary = {
        "method": (
            "two_component_logit_gmm_"
            "empirical_benign_tail"
        ),
        "benign_quantile": (
            args.benign_quantile
        ),
        "folds": 12,
        "device": str(device),
        "mean_fpr": float(
            df["fpr"].mean()
        ),
        "std_fpr": float(
            df["fpr"].std(ddof=1)
        ),
        "min_fpr": float(
            df["fpr"].min()
        ),
        "max_fpr": float(
            df["fpr"].max()
        ),
        "mean_false_alerts_per_hour": float(
            df[
                "false_alerts_per_hour"
            ].mean()
        ),
        "std_false_alerts_per_hour": float(
            df[
                "false_alerts_per_hour"
            ].std(ddof=1)
        ),
        "min_false_alerts_per_hour": float(
            df[
                "false_alerts_per_hour"
            ].min()
        ),
        "max_false_alerts_per_hour": float(
            df[
                "false_alerts_per_hour"
            ].max()
        ),
        "mean_attack_recall": float(
            df["attack_recall"].mean()
        ),
        "std_attack_recall": float(
            df["attack_recall"].std(ddof=1)
        ),
        "min_attack_recall": float(
            df["attack_recall"].min()
        ),
        "max_attack_recall": float(
            df["attack_recall"].max()
        ),
        "mean_inferred_benign_fraction": float(
            df[
                "inferred_benign_fraction"
            ].mean()
        ),
    }

    with open(
        output_root
        / "mixture_calibration_metrics.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nSummary:")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

