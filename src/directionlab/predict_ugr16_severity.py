"""Run inference with a saved UGR16 severity checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from directionlab.models import DirectionalThreatModel
from directionlab.ugr16 import CausalSequenceDataset, load_ugr16, log_standardize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="reports/ugr16_severity_full/model.pt",
    )
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument(
        "--output",
        default="reports/ugr16_severity_full/predictions.jsonl",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--history", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    train_split, test_split = load_ugr16(args.data_root)
    train_x, test_x = log_standardize(
        train_split.x,
        test_split.x,
    )

    target = np.log1p(
        test_split.attack_counts[:, 4]
    ).astype(np.float32)

    dataset = CausalSequenceDataset(
        test_x,
        target,
        args.history,
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )

    state = checkpoint["model"]

    input_dim = state["static_encoder.0.weight"].shape[1]
    hidden_dim = state["static_encoder.0.weight"].shape[0]

    model = DirectionalThreatModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        classes=2,
    )

    model.load_state_dict(state)
    model.eval()

    rows = []
    offset = 0

    with torch.no_grad():
        for current, sequence, _ in loader:
            output = model(
                current,
                sequence,
            )["severity"].numpy()

            for predicted in output:
                if len(rows) >= args.limit:
                    break

                safe_prediction = max(float(predicted), 0.0)

                rows.append(
                    {
                        "timestamp": str(
                            test_split.timestamps[offset]
                        ),
                        "predicted_log1p_blacklist_count": float(
                            predicted
                        ),
                        "predicted_blacklist_count": float(
                            np.expm1(safe_prediction)
                        ),
                        "observed_log1p_blacklist_count": float(
                            target[offset]
                        ),
                    }
                )

                offset += 1

            if len(rows) >= args.limit:
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(json.dumps(row) for row in rows)
        + "\n"
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "predictions_written": len(rows),
                "output": str(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
