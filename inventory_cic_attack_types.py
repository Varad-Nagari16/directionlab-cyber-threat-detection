from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/CIC-IDS2017/MachineLearningCVE"
OUT = ROOT / "reports/cic_attack_inventory"

OUT.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

files = sorted(DATA.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA}")

all_rows = []

for path in files:
    print(f"\nScanning: {path.name}")
    counts = Counter()
    total = 0

    # Read only the header first to locate the label column.
    header = pd.read_csv(path, nrows=0)
    label_col = next(
        (
            col for col in header.columns
            if str(col).strip().lower() == "label"
        ),
        None,
    )

    if label_col is None:
        raise ValueError(f"No Label column found in {path.name}")

    # Process just the label column in chunks.
    for chunk in pd.read_csv(
        path,
        usecols=[label_col],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        labels = (
            chunk[label_col]
            .astype("string")
            .str.strip()
            .fillna("MISSING")
        )

        counts.update(labels.tolist())
        total += len(chunk)

    print(f"  Total flows: {total:,}")

    for label, count in sorted(counts.items()):
        all_rows.append({
            "file": path.name,
            "day": path.name.split("-")[0].capitalize(),
            "label": label,
            "flows": count,
            "total_flows_in_file": total,
            "fraction_of_file": count / total if total else 0,
        })

    print("  Label counts:")
    for label, count in counts.most_common():
        print(f"    {label}: {count:,}")

result = pd.DataFrame(all_rows)

output_csv = OUT / "attack_type_inventory.csv"
result.to_csv(output_csv, index=False)

# Summarize by day and attack label.
by_day = (
    result.groupby(["day", "label"], as_index=False)["flows"]
    .sum()
    .sort_values(["day", "label"])
)

summary_csv = OUT / "attack_counts_by_day.csv"
by_day.to_csv(summary_csv, index=False)

print("\n" + "=" * 70)
print("ATTACK COUNTS BY DAY")
print("=" * 70)
print(
    by_day.pivot(
        index="label",
        columns="day",
        values="flows",
    ).fillna(0).astype(int).to_string()
)

print(f"\nSaved detailed inventory: {output_csv}")
print(f"Saved day summary:       {summary_csv}")