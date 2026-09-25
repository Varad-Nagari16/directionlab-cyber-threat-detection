from pathlib import Path

import pandas as pd
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT
    / "data/raw/CIC-IDS2017/MachineLearningCVE/"
    / "Tuesday-WorkingHours.pcap_ISCX.csv"
)

# Use the actual Tuesday filename if it differs.
PREDICTIONS = (
    ROOT
    / "reports/cic_ids2017_cross_day_tuesday/predictions.csv"
)

OUT = ROOT / "reports/cic_ids2017_attack_analysis"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading Tuesday labels...")
source = pd.read_csv(SOURCE, low_memory=False)
pred = pd.read_csv(PREDICTIONS)

# Locate the label column despite possible whitespace.
label_col = next(
    (c for c in source.columns if c.strip().lower() == "label"),
    None,
)
if label_col is None:
    raise ValueError(f"No Label column found. Columns: {list(source.columns)}")

if len(source) != len(pred):
    raise ValueError(
        f"Row count mismatch: source={len(source)}, "
        f"predictions={len(pred)}"
    )

if "y_true" not in pred or "attack_score" not in pred:
    raise ValueError(
        "Predictions must contain y_true and attack_score columns."
    )

# Confirm prediction labels agree with the original dataset's binary labels.
labels = source[label_col].astype(str).str.strip()
source_binary = (labels.str.upper() != "BENIGN").astype(int)

if not (source_binary.to_numpy() == pred["y_true"].to_numpy()).all():
    raise ValueError(
        "Prediction labels do not align with source rows. "
        "Stopping rather than producing misleading results."
    )

# Fixed threshold selected using Wednesday only.
thresholds = {
    "default_0.5": 0.5,
    "calibrated_0.04": 0.04,
}

rows = []

for attack_type, group in source.groupby(label_col, dropna=False):
    attack_type = str(attack_type).strip()
    idx = group.index

    y_true = source_binary.iloc[idx].to_numpy()
    scores = pred["attack_score"].iloc[idx].to_numpy()

    row = {
        "attack_type": attack_type,
        "flows": len(group),
        "actual_attacks": int(y_true.sum()),
    }

    for name, threshold in thresholds.items():
        y_hat = (scores >= threshold).astype(int)

        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        fn = int(((y_true == 1) & (y_hat == 0)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())

        row[f"{name}_detected"] = tp
        row[f"{name}_missed"] = fn
        row[f"{name}_recall"] = (
            tp / (tp + fn) if tp + fn else None
        )
        row[f"{name}_false_alarms"] = fp

    rows.append(row)

result = pd.DataFrame(rows)

# Put attack types with the most missed flows first.
result = result.sort_values(
    "calibrated_0.04_missed",
    ascending=False,
)

output = OUT / "attack_type_breakdown.csv"
result.to_csv(output, index=False)

print("\nAttack-type breakdown:")
print(result.to_string(index=False))
print(f"\nSaved: {output}")