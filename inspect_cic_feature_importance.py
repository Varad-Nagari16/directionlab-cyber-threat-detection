from pathlib import Path

import pandas as pd
from joblib import load


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "reports" / "cic_all_attack_coverage" / "model.joblib"
OUT_DIR = ROOT / "reports" / "cic_all_attack_coverage"

model = load(MODEL_PATH)

classifier = model.named_steps["classifier"]
feature_names = model.feature_names_in_

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": classifier.feature_importances_,
})

importance = importance.sort_values(
    "importance", ascending=False
).reset_index(drop=True)

importance["cumulative_importance"] = importance["importance"].cumsum()

out_path = OUT_DIR / "feature_importance.csv"
importance.to_csv(out_path, index=False)

print("\nTop 25 features:")
print(importance.head(25).to_string(index=False))

print("\nCumulative importance:")
for n in [5, 10, 20, 30]:
    total = importance.head(n)["importance"].sum()
    print(f"Top {n:2d}: {total:.4f} ({total:.1%})")

print(f"\nSaved: {out_path}")