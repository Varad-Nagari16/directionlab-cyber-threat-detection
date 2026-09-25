from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT = Path("data/raw/UGR16v2.X.csv")
OUT = Path("reports/ugr16_anomaly")
OUT.mkdir(parents=True, exist_ok=True)

print(f"Loading {INPUT} ...")
df = pd.read_csv(INPUT, low_memory=False)

if "Row" not in df.columns:
    raise ValueError("Expected the UGR16 timestamp column 'Row'.")

timestamps = df["Row"].astype(str)
X = df.drop(columns=["Row"]).apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = np.log1p(X.clip(lower=0))

if X.empty or len(X.columns) == 0:
    raise ValueError("No usable feature columns found.")

print(f"Rows: {len(X):,} | Features: {len(X.columns)}")
print("Fitting Isolation Forest ...")

model = IsolationForest(
    n_estimators=100,
    max_samples=min(256, len(X)),
    contamination=0.01,
    random_state=7,
    n_jobs=-1,
)
model.fit(X)

# Higher score means more anomalous.
scores = -model.decision_function(X)
flags = model.predict(X) == -1

result = pd.DataFrame({
    "timestamp": timestamps,
    "anomaly_score": scores,
    "is_anomaly": flags,
})
result.to_csv(OUT / "predictions.csv", index=False)

joblib.dump({
    "model": model,
    "features": list(X.columns),
    "transform": "log1p after clipping negative values to zero",
    "contamination": 0.01,
}, OUT / "model.joblib")

summary = {
    "dataset": "UGR16v2 feature mirror",
    "task": "unsupervised anomaly detection",
    "rows_scored": int(len(X)),
    "feature_count": int(len(X.columns)),
    "anomalies_flagged": int(flags.sum()),
    "anomaly_rate": float(flags.mean()),
    "note": "Anomaly flags are not confirmed attack labels.",
    "predictions": str(OUT / "predictions.csv"),
    "model": str(OUT / "model.joblib"),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary, indent=2))
