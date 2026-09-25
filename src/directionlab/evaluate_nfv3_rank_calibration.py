"""Evaluate unsupervised rank calibration on NFV3 host-held-out folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset


RANK_CUTOFFS = (0.90, 0.95, 0.975, 0.99, 0.995, 0.999)


EXCLUDED = {
    "timestamp", "Label", "label_binary", "source_file", "source_host_group",
    "source_host_id", "Protocol", "Attack"
}


def load_frame(
    path: Path,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, low_memory=False)

    values = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0)
        .to_numpy(dtype=np.float32)
    )

    return frame, values


def infer(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    history: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:

    dataset = CausalSequenceDataset(x, y, history)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    scores = []
    labels = []

    model.eval()

    with torch.no_grad():
        for current, sequence, target in loader:
            output = model(
                current.to(device),
                sequence.to(device),
            )

            scores.append(
                torch.sigmoid(output["binary_logit"])
                .cpu()
                .numpy()
            )

            labels.append(target.numpy())

    return np.concatenate(scores), np.concatenate(labels)


def empirical_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Convert scores into their empirical CDF rank relative to reference scores.

    No labels are used.
    """

    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)

    positions = np.searchsorted(
        reference,
        values,
        side="right",
    )

    return positions / max(len(reference), 1)


def evaluate_rank(
    labels: np.ndarray,
    ranks: np.ndarray,
    cutoff: float,
    duration_hours: float,
) -> dict[str, float]:

    predicted = ranks >= cutoff

    benign = labels == 0
    attack = labels == 1

    false_alerts = int((predicted & benign).sum())
    true_negatives = int((~predicted & benign).sum())
    true_positives = int((predicted & attack).sum())
    false_negatives = int((~predicted & attack).sum())

    benign_count = max(int(benign.sum()), 1)
    attack_count = max(int(attack.sum()), 1)

    return {
        "rank_cutoff": float(cutoff),
        "false_alerts": false_alerts,
        "true_negatives": true_negatives,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "benign_false_positive_rate": false_alerts / benign_count,
        "attack_recall": true_positives / attack_count,
        "false_alerts_per_hour": (
            false_alerts / duration_hours
            if duration_hours > 0
            else None
        ),
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold-root",
        default="data/processed/nfv3_full_host_folds",
    )

    parser.add_argument(
        "--adaptation-root",
        default="data/processed/nfv3_adaptation",
    )

    parser.add_argument(
        "--reports-root",
        default="reports/nfv3_full_cv",
    )

    parser.add_argument(
        "--output-root",
        default="reports/nfv3_rank_calibration",
    )

    parser.add_argument(
        "--history",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []

    fold_root = Path(args.fold_root)
    adaptation_root = Path(args.adaptation_root)
    reports_root = Path(args.reports_root)

    for fold_dir in sorted(fold_root.glob("fold_*")):

        fold_name = fold_dir.name

        checkpoint_path = (
            reports_root / fold_name / "model.pt"
        )

        adaptation_path = (
            adaptation_root / fold_name / "adaptation.csv"
        )

        evaluation_path = (
            adaptation_root / fold_name / "evaluation.csv"
        )

        if not checkpoint_path.exists():
            print(f"Skipping {fold_name}: missing model.pt")
            continue

        if not adaptation_path.exists() or not evaluation_path.exists():
            print(f"Skipping {fold_name}: missing adaptation/evaluation files")
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        feature_columns = checkpoint["feature_columns"]

        adaptation_frame, adaptation_x = load_frame(
            adaptation_path,
            feature_columns,
        )

        evaluation_frame, evaluation_x = load_frame(
            evaluation_path,
            feature_columns,
        )

        scaler = StandardScaler()

        scaler.mean_ = np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float64,
        )

        scaler.scale_ = np.asarray(
            checkpoint["scaler_scale"],
            dtype=np.float64,
        )

        scaler.n_features_in_ = len(feature_columns)

        adaptation_scaled = scaler.transform(
            np.log1p(adaptation_x)
        ).astype(np.float32)

        evaluation_scaled = scaler.transform(
            np.log1p(evaluation_x)
        ).astype(np.float32)

        config = checkpoint.get("config", {})

        hidden_dim = int(
            config.get("hidden_dim", 32)
        )

        model = DirectionalThreatModel(
            input_dim=len(feature_columns),
            hidden_dim=hidden_dim,
            classes=2,
        ).to(device)

        model.load_state_dict(
            checkpoint["model"]
        )

        adaptation_labels = adaptation_frame[
            "label_binary"
        ].to_numpy(dtype=np.float32)

        evaluation_labels = evaluation_frame[
            "label_binary"
        ].to_numpy(dtype=np.float32)

        adaptation_scores, _ = infer(
            model,
            adaptation_scaled,
            adaptation_labels,
            args.history,
            args.batch_size,
            device,
        )

        evaluation_scores, evaluation_labels = infer(
            model,
            evaluation_scaled,
            evaluation_labels,
            args.history,
            args.batch_size,
            device,
        )

        evaluation_ranks = empirical_rank(
            adaptation_scores,
            evaluation_scores,
        )

        start = pd.to_numeric(
            evaluation_frame["FLOW_START_MILLISECONDS"],
            errors="coerce",
        )

        end = pd.to_numeric(
            evaluation_frame["FLOW_END_MILLISECONDS"],
            errors="coerce",
        )

        valid_time = pd.concat(
            [start, end],
            axis=1,
        ).dropna()

        duration_hours = max(
            (
                valid_time.iloc[:, 1].max()
                - valid_time.iloc[:, 0].min()
            ) / 3_600_000.0,
            1e-9,
        )

        adaptation_attack_rate = float(
            adaptation_labels.mean()
        )

        for cutoff in RANK_CUTOFFS:

            metrics = evaluate_rank(
                evaluation_labels,
                evaluation_ranks,
                cutoff,
                duration_hours,
            )

            record = {
                "fold": fold_name,
                "adaptation_rows": int(len(adaptation_labels)),
                "evaluation_rows": int(len(evaluation_labels)),
                "adaptation_attack_rate_audit": adaptation_attack_rate,
                "duration_hours": float(duration_hours),
                **metrics,
            }

            records.append(record)

            print(
                json.dumps(
                    record,
                    sort_keys=True,
                )
            )

    if not records:
        raise SystemExit(
            "No folds were evaluated"
        )

    output_csv = (
        output_root / "rank_calibration_metrics.csv"
    )

    output_json = (
        output_root / "rank_calibration_metrics.json"
    )

    pd.DataFrame(records).to_csv(
        output_csv,
        index=False,
    )

    output_json.write_text(
        json.dumps(
            records,
            indent=2,
        )
        + "\n"
    )

    print(
        json.dumps(
            {
                "status": "complete",
                "folds": len(
                    sorted(
                        set(
                            r["fold"]
                            for r in records
                        )
                    )
                ),
                "rows": len(records),
                "output": str(output_csv),
                "device": str(device),
                "note": (
                    "Adaptation labels are used only for audit; "
                    "rank calibration itself uses scores only."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
