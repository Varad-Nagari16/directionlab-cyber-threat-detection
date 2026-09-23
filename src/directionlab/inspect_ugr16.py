"""Inspect the downloaded UGR16 feature mirror without modifying source files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"


def inspect(name: str) -> None:
    path = DATA / name
    frame = pd.read_csv(path)
    print({"file": name, "shape": frame.shape, "dtypes": frame.dtypes.astype(str).value_counts().to_dict()})
    print(frame.head(2).to_string(index=False, header=False))
    print({"missing": int(frame.isna().sum().sum()), "unique_rows": int(len(frame.drop_duplicates()))})


def main() -> None:
    for name in ("UGR16v2.X.csv", "UGR16v2.Y.csv", "UGR16v2.test.X.csv", "UGR16v2.test.Y.csv"):
        inspect(name)

    labels = pd.read_csv(DATA / "UGR16v2.Y.csv")
    numeric = labels.drop(columns=["Row"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    print({"label_columns": list(numeric.columns)})
    print({"label_column_sums": numeric.sum(axis=0).round(3).to_dict()})
    print({"positive_rows": int((numeric.sum(axis=1) > 0).sum()), "negative_rows": int((numeric.sum(axis=1) == 0).sum()), "total_rows": len(labels)})

    test_labels = pd.read_csv(DATA / "UGR16v2.test.Y.csv")
    test_numeric = test_labels.drop(columns=["Row"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    print({"test_positive_rows": int((test_numeric.sum(axis=1) > 0).sum()), "test_negative_rows": int((test_numeric.sum(axis=1) == 0).sum()), "test_total_rows": len(test_labels)})


if __name__ == "__main__":
    main()
