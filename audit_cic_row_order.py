from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "CIC-IDS2017" / "MachineLearningCVE"
OUT_DIR = ROOT / "reports" / "cic_row_order_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


def find_column(columns, wanted):
    """Find a column while ignoring surrounding whitespace and case."""
    wanted = wanted.strip().lower()
    for col in columns:
        if str(col).strip().lower() == wanted:
            return col
    return None


def inspect_file(path):
    print(f"\nAuditing: {path.name}")

    # CIC files may contain a UTF-8 BOM.
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        header = pd.read_csv(path, nrows=0, encoding="cp1252")
        encoding = "cp1252"

    label_col = find_column(header.columns, "Label")
    time_col = find_column(header.columns, "Timestamp")

    if label_col is None:
        raise ValueError(f"No Label column found in {path.name}")

    usecols = [label_col]
    if time_col is not None:
        usecols.append(time_col)

    # Read only labels and timestamps, not all 77 features.
    parts = []
    offset = 0

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        encoding=encoding,
        low_memory=False,
    ):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        label_name = find_column(chunk.columns, "Label")
        time_name = find_column(chunk.columns, "Timestamp")

        labels = chunk[label_name].astype("string").str.strip()
        block = pd.DataFrame({
            "row": range(offset, offset + len(chunk)),
            "label": labels,
        })

        if time_name is not None:
            block["timestamp"] = chunk[time_name].astype("string").str.strip()

        parts.append(block)
        offset += len(chunk)

    data = pd.concat(parts, ignore_index=True)
    n = len(data)

    # Counts in sequential thirds: this is diagnostic, not a proposed split.
    data["third"] = pd.cut(
        data["row"],
        bins=[-1, n / 3, 2 * n / 3, n],
        labels=["first_third", "middle_third", "last_third"],
    )

    thirds = (
        data.groupby(["third", "label"], observed=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    thirds.insert(0, "file", path.name)

    # Count label transitions, which reveals whether labels appear in blocks.
    changed = data["label"].ne(data["label"].shift()).fillna(True)
    run_starts = data.loc[changed, ["row", "label"]].copy()
    run_starts["run_end"] = (
        run_starts["row"].shift(-1).fillna(n).astype(int) - 1
    )
    run_starts["run_length"] = run_starts["run_end"] - run_starts["row"] + 1
    run_starts.insert(0, "file", path.name)

    # Compact summary for each file.
    summary = {
        "file": path.name,
        "rows": n,
        "label_column": label_col,
        "timestamp_column_found": time_col is not None,
        "distinct_labels": data["label"].nunique(dropna=False),
        "label_runs": len(run_starts),
        "longest_label_run": int(run_starts["run_length"].max()),
        "first_label": str(data["label"].iloc[0]),
        "last_label": str(data["label"].iloc[-1]),
    }

    if "timestamp" in data.columns:
        ts = pd.to_datetime(data["timestamp"], errors="coerce")
        summary["timestamp_parse_rate"] = round(float(ts.notna().mean()), 4)
        summary["first_valid_timestamp"] = (
            str(ts.dropna().iloc[0]) if ts.notna().any() else ""
        )
        summary["last_valid_timestamp"] = (
            str(ts.dropna().iloc[-1]) if ts.notna().any() else ""
        )
        summary["timestamps_monotonic"] = bool(
            ts.dropna().is_monotonic_increasing
        )
    else:
        summary["timestamp_parse_rate"] = None
        summary["first_valid_timestamp"] = ""
        summary["last_valid_timestamp"] = ""
        summary["timestamps_monotonic"] = None

    print(
        f"  rows={n:,} | labels={summary['distinct_labels']} "
        f"| label runs={summary['label_runs']:,} "
        f"| longest run={summary['longest_label_run']:,}"
    )

    return summary, thirds, run_starts


def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    summaries = []
    thirds_all = []
    runs_all = []

    for path in files:
        summary, thirds, runs = inspect_file(path)
        summaries.append(summary)
        thirds_all.append(thirds)
        runs_all.append(runs)

    pd.DataFrame(summaries).to_csv(
        OUT_DIR / "file_summary.csv", index=False
    )
    pd.concat(thirds_all, ignore_index=True).to_csv(
        OUT_DIR / "label_counts_by_third.csv", index=False
    )
    pd.concat(runs_all, ignore_index=True).to_csv(
        OUT_DIR / "label_runs.csv", index=False
    )

    print("\nAudit complete.")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()