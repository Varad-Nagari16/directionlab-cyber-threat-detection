"""Direction-preserving data contract and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "record_id",
    "timestamp",
    "duration_s",
    "bytes",
    "packets",
    "transport_protocol",
    "src_port",
    "dst_port",
    "label",
}

# These names are deliberately broad so a source adapter fails closed.
FORBIDDEN_TOKENS = (
    "reverse",
    "rev_",
    "bidir",
    "bidirectional",
    "backward",
    "payload",
    "future",
    "session_total",
)


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    missing_required: tuple[str, ...]
    duplicate_record_ids: int
    invalid_numeric_rows: int
    forbidden_columns: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_required
            or self.duplicate_record_ids
            or self.invalid_numeric_rows
            or self.forbidden_columns
        )

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ValueError(self.summary())

    def summary(self) -> str:
        return (
            "Invalid direction-preserving table: "
            f"missing={self.missing_required}, "
            f"duplicate_ids={self.duplicate_record_ids}, "
            f"invalid_numeric_rows={self.invalid_numeric_rows}, "
            f"forbidden_columns={self.forbidden_columns}"
        )


def _forbidden_columns(columns: Iterable[str]) -> tuple[str, ...]:
    found = []
    for column in columns:
        lowered = column.lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            found.append(column)
    return tuple(sorted(found))


def validate_frame(frame: pd.DataFrame) -> ValidationReport:
    """Validate a canonical one-way flow table without changing it."""
    missing = tuple(sorted(REQUIRED_COLUMNS - set(frame.columns)))
    duplicate_ids = (
        int(frame["record_id"].duplicated().sum()) if "record_id" in frame else 0
    )
    numeric_columns = ["duration_s", "bytes", "packets"]
    invalid_numeric = 0
    if all(column in frame.columns for column in numeric_columns):
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        invalid_numeric = int((numeric.isna().any(axis=1) | (numeric < 0).any(axis=1)).sum())

    return ValidationReport(
        rows=len(frame),
        missing_required=missing,
        duplicate_record_ids=duplicate_ids,
        invalid_numeric_rows=invalid_numeric,
        forbidden_columns=_forbidden_columns(frame.columns),
    )

