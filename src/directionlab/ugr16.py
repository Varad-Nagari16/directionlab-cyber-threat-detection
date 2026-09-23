"""UGR'16 feature-mirror adapter.

This adapter targets the public one-minute feature mirror, not the original
raw NetFlow release. It keeps the timestamp key for ordering and removes it
from model inputs. The proxy is useful for a first reproducible run but does
not replace the stricter raw one-way flow experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


ATTACK_LABELS = (
    "labeldos",
    "labelscan11",
    "labelscan44",
    "labelnerisbotnet",
    "labelblacklist",
    "labelanomalyidpscan",
    "labelanomalysshscan",
    "labelanomalyspam",
)


@dataclass(frozen=True)
class UGRSplit:
    x: np.ndarray
    y_binary: np.ndarray
    timestamps: np.ndarray
    attack_counts: np.ndarray


class CausalSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, history: int) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.history = history

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = max(0, index - self.history + 1)
        sequence = np.zeros((self.history, self.x.shape[1]), dtype=np.float32)
        chunk = self.x[start : index + 1]
        sequence[-len(chunk) :] = chunk
        current = self.x[index]
        return torch.from_numpy(current), torch.from_numpy(sequence), torch.tensor(self.y[index])


def _read_pair(root: Path, prefix: str) -> UGRSplit:
    x_path = root / f"{prefix}.X.csv"
    y_path = root / f"{prefix}.Y.csv"
    x = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    if not x.index.equals(y.index):
        raise ValueError(f"Row keys do not align for {prefix}")
    missing = [label for label in ATTACK_LABELS if label not in y.columns]
    if missing:
        raise ValueError(f"Missing UGR16 labels: {missing}")
    attack_counts = y.loc[:, ATTACK_LABELS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    return UGRSplit(
        x=x.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32),
        y_binary=(attack_counts.sum(axis=1) > 0).astype(np.float32),
        timestamps=x.index.to_numpy(),
        attack_counts=attack_counts,
    )


def load_ugr16(root: str | Path) -> tuple[UGRSplit, UGRSplit]:
    """Load the downloaded mirror files after normalizing their headers."""
    root = Path(root)
    mapping = {
        "UGR16v2.Xtrain.csv": "UGR16v2.X.csv",
        "UGR16v2.Ytrain.csv": "UGR16v2.Y.csv",
        "UGR16v2.Xtest.csv": "UGR16v2.test.X.csv",
        "UGR16v2.Ytest.csv": "UGR16v2.test.Y.csv",
    }
    for source, target in mapping.items():
        source_path, target_path = root / source, root / target
        if not target_path.exists():
            source_path.rename(target_path)
    return _read_pair(root, "UGR16v2"), _read_pair(root, "UGR16v2.test")


def log_standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Fit preprocessing on train only and apply it to later partitions."""
    train_log = np.log1p(np.clip(train, a_min=0, a_max=None))
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_log).astype(np.float32)
    result = [train_scaled]
    for other in others:
        other_log = np.log1p(np.clip(other, a_min=0, a_max=None))
        result.append(scaler.transform(other_log).astype(np.float32))
    return tuple(result)


def add_time_features(features: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Append causal calendar features derived from each observed timestamp."""
    parsed = pd.to_datetime(pd.Series(timestamps.astype(str)), format="%Y%m%d%H%M", errors="coerce")
    if parsed.isna().any():
        raise ValueError("UGR16 timestamps must use YYYYMMDDHHMM format")
    minute = (parsed.dt.hour * 60 + parsed.dt.minute).to_numpy(dtype=np.float32)
    weekday = parsed.dt.dayofweek.to_numpy(dtype=np.float32)
    minute_angle = 2 * np.pi * minute / (24 * 60)
    weekday_angle = 2 * np.pi * weekday / 7
    calendar = np.column_stack(
        [
            np.sin(minute_angle),
            np.cos(minute_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            (weekday >= 5).astype(np.float32),
        ]
    ).astype(np.float32)
    return np.concatenate([features.astype(np.float32), calendar], axis=1)
