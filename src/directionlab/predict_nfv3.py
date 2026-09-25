
"""Run a trained DirectionLab NFV3 detector on a CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    """Load and validate a saved model checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    required = {
        "model",
        "config",
        "feature_columns",
        "scaler_mean",
        "scaler_scale",
        "results",
    }

    missing = required - set(checkpoint.keys())

    if missing:
        raise ValueError(
            f"Checkpoint is missing fields: {sorted(missing)}"
        )

    return checkpoint


def load_input(
    path: Path,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load CSV and extract the model's expected features."""

    frame = pd.read_csv(path, low_memory=False)

    if frame.empty:
        raise ValueError(f"Input CSV contains no rows: {path}")

    missing = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Input CSV is missing required model features: "
            + ", ".join(missing)
        )

    x = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0)
        .to_numpy(dtype=np.float32)
    )

    return frame, x


def transform(
    x: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """Apply log1p and the saved training scaler."""

    x_log = np.log1p(x)

    safe_scale = np.where(
        scaler_scale == 0,
        1.0,
        scaler_scale,
    )

    return ((x_log - scaler_mean) / safe_scale).astype(
        np.float32
    )


def predict(
    model: torch.nn.Module,
    x: np.ndarray,
    history: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate one threat score for every input row."""

    if history <= 0:
        raise ValueError("History must be greater than zero.")

    dummy_labels = np.zeros(len(x), dtype=np.float32)

    dataset = CausalSequenceDataset(
        x,
        dummy_labels,
        history,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    scores = []

    model.eval()

    with torch.no_grad():
        for current, history_batch, _ in loader:
            output = model(
                current.to(device),
                history_batch.to(device),
            )

            probabilities = torch.sigmoid(
                output["binary_logit"]
            )

            scores.append(probabilities.cpu().numpy())

    if not scores:
        raise ValueError("The model produced no predictions.")

    return np.concatenate(scores).astype(np.float32)


def run_prediction(
    input_path: str | Path,
    checkpoint_path: str | Path = (
        "reports/nfv3_full_cv/fold_01/model.pt"
    ),
    output_path: str | Path = "reports/nfv3_prediction.csv",
    batch_size: int = 512,
    history: int | None = None,
) -> dict:
    """
    Run inference on a CSV, save row-level predictions,
    and return a JSON-serializable summary.

    This function can be called by the CLI or by an API.
    """

    input_path = Path(input_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = load_checkpoint(checkpoint_path, device)

    feature_columns = checkpoint["feature_columns"]
    config = checkpoint["config"]

    history_length = (
        int(history)
        if history is not None
        else int(config.get("history", 32))
    )

    hidden_dim = int(config.get("hidden_dim", 32))

    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    scaler_scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float32,
    )

    if len(scaler_mean) != len(feature_columns):
        raise ValueError(
            "Saved scaler mean length does not match "
            "the checkpoint feature count."
        )

    if len(scaler_scale) != len(feature_columns):
        raise ValueError(
            "Saved scaler scale length does not match "
            "the checkpoint feature count."
        )

    results = checkpoint["results"]

    if "calibrated_threshold" not in results:
        raise ValueError(
            "Checkpoint does not contain a calibrated threshold."
        )

    threshold = float(results["calibrated_threshold"])

    # Load and transform input using the saved training setup.
    frame, x = load_input(input_path, feature_columns)
    input_rows = len(frame)

    x = transform(x, scaler_mean, scaler_scale)

    # Construct the model and load its trained weights.
    model = DirectionalThreatModel(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        classes=2,
    ).to(device)

    model.load_state_dict(checkpoint["model"])

    # Run exactly one inference pass.
    scores = predict(
        model=model,
        x=x,
        history=history_length,
        batch_size=batch_size,
        device=device,
    )

    if len(scores) != input_rows:
        raise RuntimeError(
            f"Got {len(scores)} predictions for "
            f"{input_rows} input rows."
        )

    predictions = (scores >= threshold).astype(np.int64)

    # Keep every original row and its corresponding prediction.
    output_frame = frame.copy()
    output_frame["threat_score"] = scores
    output_frame["prediction"] = np.where(
        predictions == 1,
        "ATTACK",
        "BENIGN",
    )
    output_frame["threshold"] = threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)

    attack_count = int(predictions.sum())
    benign_count = int(len(predictions) - attack_count)

    summary = {
        "status": "complete",
        "device": str(device),
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "input_rows": int(input_rows),
        "prediction_rows": int(len(scores)),
        "feature_count": int(len(feature_columns)),
        "history": int(history_length),
        "threshold": threshold,
        "attack_predictions": attack_count,
        "benign_predictions": benign_count,
        "attack_rate": float(attack_count / len(scores)),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DirectionLab NFV3 detector on a CSV."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--checkpoint",
        default="reports/nfv3_full_cv/fold_01/model.pt",
    )
    parser.add_argument(
        "--output",
        default="reports/nfv3_prediction.csv",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=None)

    args = parser.parse_args()

    summary = run_prediction(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        history=args.history,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()