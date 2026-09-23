"""Current-flow feature construction.

Every feature here is computed from the observed row only. Learned preprocessing
such as scaling must be fitted on the training split by the caller.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


NUMERIC_FEATURES = (
    "duration_s",
    "bytes",
    "packets",
    "bytes_per_s",
    "packets_per_s",
    "src_port",
    "dst_port",
)
CATEGORICAL_FEATURES = ("transport_protocol",)


def build_current_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an allowlisted feature matrix for observed directional rows."""
    required = {"duration_s", "bytes", "packets", "src_port", "dst_port", "transport_protocol"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    result = pd.DataFrame(index=frame.index)
    duration = pd.to_numeric(frame["duration_s"], errors="coerce").fillna(0.0).clip(lower=0.0)
    safe_duration = duration.where(duration > 0, 1.0)
    result["duration_s"] = duration.astype(np.float32)
    result["bytes"] = pd.to_numeric(frame["bytes"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(np.float32)
    result["packets"] = pd.to_numeric(frame["packets"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(np.float32)
    result["bytes_per_s"] = (result["bytes"] / safe_duration).astype(np.float32)
    result["packets_per_s"] = (result["packets"] / safe_duration).astype(np.float32)
    result["src_port"] = pd.to_numeric(frame["src_port"], errors="coerce").fillna(0).clip(0, 65535).astype(np.float32)
    result["dst_port"] = pd.to_numeric(frame["dst_port"], errors="coerce").fillna(0).clip(0, 65535).astype(np.float32)
    protocol = frame["transport_protocol"].astype("string").fillna("unknown")
    result["protocol_tcp"] = protocol.str.lower().eq("tcp").astype(np.float32)
    result["protocol_udp"] = protocol.str.lower().eq("udp").astype(np.float32)
    result["protocol_other"] = (~protocol.str.lower().isin({"tcp", "udp"})).astype(np.float32)
    return result


def make_causal_windows(
    features: np.ndarray,
    labels: np.ndarray,
    history: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Create left-padded causal windows ending at each row.

    Window ``i`` contains rows no later than ``i``. Zero padding is used before
    the beginning of the stream and is not populated from future rows.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [rows, features]")
    if len(features) != len(labels):
        raise ValueError("features and labels must have equal length")
    if history < 1:
        raise ValueError("history must be positive")

    windows = np.zeros((len(features), history, features.shape[1]), dtype=np.float32)
    for end in range(len(features)):
        start = max(0, end - history + 1)
        chunk = features[start : end + 1]
        windows[end, -len(chunk) :] = chunk
    return windows, np.asarray(labels)
