
"""DirectionLab dataset detection and labeled-CSV preparation."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Task = Literal["binary", "multiclass"]


def normalize_name(value: object) -> str:
    text = str(value).replace("\ufeff", "").strip().lower()
    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^a-z0-9]+", "_", text),
    ).strip("_")


def read_header(path: str | Path) -> list[str]:
    """Read CSV headers; let pandas disambiguate repeated column names."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        columns = pd.read_csv(
            path, nrows=0, encoding="utf-8-sig"
        ).columns
    except UnicodeDecodeError:
        columns = pd.read_csv(
            path, nrows=0, encoding="latin-1"
        ).columns

    columns = [str(column).strip() for column in columns]

    if not columns or all(not column for column in columns):
        raise ValueError("Could not find a valid header row.")

    # pandas renames repeated raw headers, e.g.:
    # "Fwd Header Length" -> "Fwd Header Length.1".
    # Treat those generated ".N" suffixes as duplicate-header markers,
    # not as a fatal normalized-name collision. The adapter later checks
    # whether known duplicated feature columns are actually identical.
    normalized = [normalize_name(column) for column in columns]
    base_names = {
        normalize_name(re.sub(r"\.\d+$", "", column))
        for column in columns
    }

    duplicates = []
    seen = set()
    for column, normalized_name in zip(columns, normalized):
        if not normalized_name:
            continue
        is_pandas_duplicate = bool(re.search(r"\.\d+$", column)) and (
            normalize_name(re.sub(r"\.\d+$", "", column)) in base_names
        )
        if normalized_name in seen and not is_pandas_duplicate:
            duplicates.append(normalized_name)
        seen.add(normalized_name)

    if duplicates:
        raise ValueError(
            f"Column names collide after normalization: {sorted(set(duplicates))}"
        )

    return columns


LABEL_ALIASES = {
    "label", "labels", "label_binary", "class", "class_label", "attack",
    "attack_cat", "attack_category", "category", "target", "is_attack",
    "labeldos", "labelscan11", "labelscan44", "labelnerisbotnet",
    "labelblacklist", "labelanomalyidpscan", "labelanomalysshscan",
    "labelanomalyspam",
}

ID_ALIASES = {
    "flow_id", "flowid", "src_ip", "source_ip", "dst_ip",
    "destination_ip", "srcip", "dstip", "source", "destination",
    "timestamp", "time", "date", "row", "uid", "id", "source_file",
    "source_host_id", "source_host_group",
}

CIC_SIGNATURE = {
    "flow_duration", "total_fwd_packets", "total_backward_packets"
}

UNSW_SIGNATURE = {"srcip", "dstip", "sport", "dport", "proto"}

NFV3_FEATURES = {
    "protocol", "l7_proto", "flow_duration_milliseconds", "min_ttl",
    "max_ttl", "longest_flow_pkt", "shortest_flow_pkt", "min_ip_pkt_len",
    "max_ip_pkt_len", "src_to_dst_second_bytes",
    "src_to_dst_avg_throughput", "num_pkts_up_to_128_bytes",
    "num_pkts_128_to_256_bytes", "num_pkts_256_to_512_bytes",
    "num_pkts_512_to_1024_bytes", "num_pkts_1024_to_1514_bytes",
    "tcp_win_max_in", "icmp_type", "icmp_ipv4_type", "dns_query_id",
    "dns_query_type", "dns_ttl_answer", "ftp_command_ret_code",
    "src_to_dst_iat_min", "src_to_dst_iat_max", "src_to_dst_iat_avg",
    "src_to_dst_iat_stddev", "flow_start_milliseconds",
    "flow_end_milliseconds",
}


@dataclass(frozen=True)
class DatasetProfile:
    detected_dataset: str
    detection_basis: str
    column_count: int
    columns: list[str]
    label_columns: list[str]
    compatible_schema: bool
    notes: list[str]


def detect_dataset(columns: list[str]) -> DatasetProfile:
    mapping = {normalize_name(c): c for c in columns}
    names = set(mapping)

    labels = [
        original
        for normalized, original in mapping.items()
        if normalized in LABEL_ALIASES
        or normalized.startswith("label_")
    ]

    if NFV3_FEATURES.issubset(names):
        family = "NF-UNSW-NB15-v3"
        basis = "all 29 NFv3 features matched"
        compatible = True
        notes = [
            "Schema match only; confirm preprocessing compatibility."
        ]

    elif CIC_SIGNATURE.issubset(names):
        family = "CICFlowMeter-compatible (CIC-IDS2017/2018-like)"
        basis = "CICFlowMeter flow and packet columns matched"
        compatible = False
        notes = [
            "Column names alone cannot distinguish CIC-IDS2017 "
            "from CIC-IDS2018."
        ]

    elif UNSW_SIGNATURE.issubset(names):
        family = "UNSW-NB15-like"
        basis = "IP, port, and protocol columns matched"
        compatible = False
        notes = ["Check labels and categorical feature handling."]

    elif {"srcipprivate", "dporthttp", "protocoltcp"} & names:
        family = "UGR16-derived / one-hot flow data"
        basis = "UGR16-like columns matched"
        compatible = False
        notes = ["Heuristic only; does not prove dataset origin."]

    else:
        family = "Unknown / custom tabular data"
        basis = "no known signature matched"
        compatible = False
        notes = [
            "Review columns and configure an explicit adapter if appropriate."
        ]

    if not labels:
        notes.append(
            "No recognized label column; supervised training is unavailable."
        )

    return DatasetProfile(
        family, basis, len(columns), columns, labels,
        compatible, notes
    )


def inspect_dataset(path: str | Path) -> dict:
    return asdict(detect_dataset(read_header(path)))


def _label_column(
    frame: pd.DataFrame,
    requested: str | None,
) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Label column {requested!r} is absent.")
        return requested

    mapping = {normalize_name(c): c for c in frame.columns}

    for key in (
        "label_binary", "label", "labels", "class", "class_label",
        "attack_cat", "attack_category", "attack", "category",
        "target", "is_attack",
    ):
        if key in mapping:
            return mapping[key]

    candidates = [
        c for c in frame.columns
        if normalize_name(c).startswith("label_")
    ]

    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        raise ValueError(
            f"Multiple label columns found; specify one: {candidates}"
        )

    raise ValueError(
        "No recognized label column. Use anomaly detection for unlabeled data."
    )


def normalize_attack_label(value: object) -> str | None:
    if pd.isna(value):
        return None

    label = normalize_name(value)

    if not label or label in {
        "nan", "none", "null", "unknown", "undefined"
    }:
        return None

    if label in {
        "benign", "normal", "normal_traffic", "non_attack",
        "nonattack", "false", "clean",
    }:
        return "BENIGN"

    return label.upper()


def prepare_labeled_csv(
    path: str | Path,
    *,
    task: Task = "binary",
    label_column: str | None = None,
    drop_columns: list[str] | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return numeric features, encoded targets, and metadata.

    Binary: BENIGN=0, any other non-missing label=1.
    Multiclass: BENIGN and each distinct attack label get stable sorted IDs.

    Fit imputers/scalers/encoders on the training split only, not here.
    """
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be 'binary' or 'multiclass'")

    path = Path(path)
    frame = pd.read_csv(path, low_memory=False, nrows=max_rows)

    if frame.empty:
        raise ValueError("No data rows found.")

    label_col = _label_column(frame, label_column)
    labels = frame[label_col].map(normalize_attack_label)
    valid = labels.notna()

    frame = frame.loc[valid].copy()
    labels = labels.loc[valid]

    if frame.empty:
        raise ValueError(f"No usable labels in {label_col!r}.")

    excluded = (
        set(LABEL_ALIASES)
        | set(ID_ALIASES)
        | {normalize_name(label_col)}
    )
    excluded.update(normalize_name(c) for c in (drop_columns or []))

    candidates = [
        c for c in frame.columns
        if normalize_name(c) not in excluded
        and not normalize_name(c).startswith("label_")
    ]

    if not candidates:
        raise ValueError("No candidate feature columns remain.")

    numeric = frame[candidates].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    # CIC-IDS2017 contains a duplicated "Fwd Header Length" column.
    # pandas disambiguates the repeated header as "Fwd Header Length.1".
    # Drop it only when its values exactly match the original column.
    original_col = next(
        (c for c in numeric.columns if str(c).strip() == "Fwd Header Length"),
        None,
    )
    duplicate_col = next(
        (c for c in numeric.columns if str(c).strip() == "Fwd Header Length.1"),
        None,
    )

    if (
        original_col is not None
        and duplicate_col is not None
        and numeric[original_col].equals(numeric[duplicate_col])
    ):
        numeric = numeric.drop(columns=[duplicate_col])

    if numeric.shape[1] == 0:
        raise ValueError(
            "No numeric features found; categorical encoding is needed."
        )

    if task == "binary":
        y = (labels != "BENIGN").astype(np.int64).to_numpy()
        class_names = ["BENIGN", "ATTACK"]
        counts = pd.Series(
            np.where(y == 0, "BENIGN", "ATTACK")
        ).value_counts().to_dict()

    else:
        class_names = sorted(
            labels.unique(),
            key=lambda x: (x != "BENIGN", x),
        )
        ids = {name: i for i, name in enumerate(class_names)}
        y = labels.map(ids).astype(np.int64).to_numpy()
        counts = labels.value_counts().to_dict()

    metadata = {
        "source_path": str(path),
        "detected_dataset": detect_dataset(
            read_header(path)
        ).detected_dataset,
        "label_column": label_col,
        "task": task,
        "rows_used": int(len(numeric)),
        "feature_columns": list(numeric.columns),
        "feature_count": int(numeric.shape[1]),
        "class_names": class_names,
        "class_counts": {
            str(k): int(v) for k, v in counts.items()
        },
        "excluded_columns": [
            c for c in frame.columns
            if c not in numeric.columns and c != label_col
        ],
        "warning": (
            "Split by time/host/source before fitting preprocessing "
            "or evaluating to reduce leakage."
        ),
    }

    return numeric, y, metadata