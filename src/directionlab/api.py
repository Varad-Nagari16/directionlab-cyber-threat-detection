from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from directionlab.dataset_support import inspect_csv
from directionlab.predict_nfv3 import run_prediction


app = FastAPI(
    title="DirectionLab Cyber Threat Detection API",
    version="1.1.0",
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "reports"
    / "nfv3_full_cv"
    / "fold_01"
    / "model.pt"
)

UGR16_MODEL = (
    PROJECT_ROOT
    / "reports"
    / "ugr16_anomaly"
    / "model.joblib"
)

FRONTEND_FILE = (
    Path(__file__).resolve().parent
    / "static"
    / "index.html"
)

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 100 MB


# Temporary directory for uploaded files and predictions.
_TEMP_DIR = TemporaryDirectory(prefix="directionlab_api_")
RESULTS_DIR = Path(_TEMP_DIR.name)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.get("/app", response_class=HTMLResponse)
def frontend():
    if not FRONTEND_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend file not found.",
        )

    return FRONTEND_FILE.read_text(encoding="utf-8")


# --------------------------------------------------
# General API routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "DirectionLab Cyber Threat Detection API",
        "status": "running",
        "version": app.version,
        "docs": "/docs",
        "dashboard": "/app",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint_exists": DEFAULT_CHECKPOINT.is_file(),
        "frontend_exists": FRONTEND_FILE.is_file(),
    }


# --------------------------------------------------
# Upload validation
# --------------------------------------------------

async def save_uploaded_csv(file: UploadFile, input_path: Path):
    """
    Save an uploaded CSV while enforcing the upload-size limit.

    Returns the number of bytes written.
    """

    total_bytes = 0

    with input_path.open("wb") as destination:
        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            total_bytes += len(chunk)

            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="CSV exceeds the 100 MB upload limit.",
                )

            destination.write(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV is empty.",
        )

    return total_bytes


def inspect_uploaded_dataset(input_path: Path) -> dict:
    """
    Detect the uploaded dataset using its CSV column names.

    This inspects the schema. It does not train a model or
    convert an incompatible dataset into NFv3 features.
    """

    try:
        inspection = inspect_csv(str(input_path))

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not inspect the uploaded CSV.",
                "reason": str(exc),
            },
        ) from exc

    if not isinstance(inspection, dict):
        raise HTTPException(
            status_code=500,
            detail="Dataset inspection returned an unexpected result.",
        )

    return inspection


def run_ugr16_anomaly_prediction(
    input_path: Path,
    output_path: Path,
) -> dict:
    """Score UGR16 rows using the saved unsupervised detector."""

    if not UGR16_MODEL.is_file():
        raise HTTPException(
            status_code=500,
            detail={
                "message": "UGR16 anomaly model not found.",
                "expected_model": str(UGR16_MODEL),
            },
        )

    saved = joblib.load(UGR16_MODEL)
    model = saved["model"]
    features = saved["features"]

    df = pd.read_csv(input_path, low_memory=False)

    if "Row" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="UGR16 CSV must contain the Row timestamp column.",
        )

    missing = [name for name in features if name not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "UGR16 feature columns do not match the model.",
                "missing_features": missing,
            },
        )

    # Preserve the exact feature order used during training.
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = np.log1p(X.clip(lower=0))

    scores = -model.decision_function(X)
    flags = model.predict(X) == -1

    results = pd.DataFrame({
        "timestamp": df["Row"].astype(str),
        "anomaly_score": scores,
        "is_anomaly": flags,
    })

    results.to_csv(output_path, index=False)

    return {
        "task": "unsupervised_anomaly_detection",
        "rows_scored": int(len(results)),
        "features_used": len(features),
        "anomalies_flagged": int(flags.sum()),
        "anomaly_rate": float(flags.mean()),
        "note": (
            "Anomaly flags indicate unusual patterns, "
            "not confirmed cyberattacks."
        ),
    }

# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detect the CSV schema and run the NFv3 model only when
    the uploaded file matches the required NFv3 feature schema.
    """

    # Validate the uploaded filename.
    if (
        not file.filename
        or Path(file.filename).suffix.lower() != ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .csv file.",
        )

    result_id = uuid4().hex

    input_path = RESULTS_DIR / f"{result_id}_input.csv"
    output_path = RESULTS_DIR / f"{result_id}_predictions.csv"

    try:
        # ------------------------------------------
        # 1. Save and validate the uploaded CSV
        # ------------------------------------------

        await save_uploaded_csv(file, input_path)

        # ------------------------------------------
        # 2. Detect the dataset format
        # ------------------------------------------

        dataset_info = inspect_uploaded_dataset(input_path)

        detected_dataset = dataset_info.get(
            "detected_dataset",
            "Unknown dataset",
        )

        is_compatible = dataset_info.get(
            "compatible_with_current_nfv3_schema",
            False,
        )

        # ------------------------------------------
        # 3. Route UGR16 to its own anomaly detector
        # ------------------------------------------

        detection_basis = dataset_info.get("detection_basis", "")
        is_ugr16 = (
            "UGR16" in str(detected_dataset).upper()
            or "UGR16" in str(detection_basis).upper()
        )

        if is_ugr16:
            summary = run_ugr16_anomaly_prediction(
                input_path=input_path,
                output_path=output_path,
            )

            return {
                "status": "complete",
                "result_id": result_id,
                "dataset": {
                    "detected_dataset": detected_dataset,
                    "detection_basis": detection_basis,
                    "task": "unsupervised anomaly detection",
                    "compatible_with_current_nfv3_schema": False,
                },
                "summary": summary,
                "download_url": f"/results/{result_id}",
            }

        # ------------------------------------------
        # 4. Reject unsupported feature schemas
        # ------------------------------------------

        if not is_compatible:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "This CSV cannot be processed by the current "
                        "NF-UNSW-NB15-v3 model."
                    ),
                    "detected_dataset": detected_dataset,
                    "detection_basis": dataset_info.get(
                        "detection_basis"
                    ),
                    "compatible_with_current_nfv3_schema": False,
                    "nfv3_feature_matches": dataset_info.get(
                        "nfv3_feature_matches"
                    ),
                    "nfv3_required_feature_count": dataset_info.get(
                        "nfv3_required_feature_count"
                    ),
                    "missing_nfv3_features": dataset_info.get(
                        "missing_nfv3_features",
                        [],
                    ),
                    "next_step": (
                        "Upload a CSV containing the required "
                        "NF-UNSW-NB15-v3 features, or use a model "
                        "trained for this dataset."
                    ),
                },
            )

        # ------------------------------------------
        # 5. Confirm NFv3 checkpoint exists
        # ------------------------------------------

        if not DEFAULT_CHECKPOINT.is_file():
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "NFv3 model checkpoint not found.",
                    "checkpoint": str(DEFAULT_CHECKPOINT),
                },
            )

        # ------------------------------------------
        # 6. Run the existing NFv3 model
        # ------------------------------------------

        summary = run_prediction(
            input_path=str(input_path),
            checkpoint_path=str(DEFAULT_CHECKPOINT),
            output_path=str(output_path),
        )

        # ------------------------------------------
        # 7. Return prediction results
        # ------------------------------------------

        return {
            "status": "complete",
            "result_id": result_id,
            "dataset": {
                "detected_dataset": detected_dataset,
                "detection_basis": dataset_info.get(
                    "detection_basis"
                ),
                "compatible_with_current_nfv3_schema": True,
                "nfv3_feature_matches": dataset_info.get(
                    "nfv3_feature_matches"
                ),
                "nfv3_required_feature_count": dataset_info.get(
                    "nfv3_required_feature_count"
                ),
            },
            "summary": summary,
            "download_url": f"/results/{result_id}",
        }

    except HTTPException:
        # Preserve expected HTTP status codes and error details.
        output_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        # Remove incomplete prediction output.
        output_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed.",
                "reason": str(exc),
            },
        ) from exc

    finally:
        await file.close()
        input_path.unlink(missing_ok=True)


# --------------------------------------------------
# Download prediction results
# --------------------------------------------------

@app.get("/results/{result_id}")
def download_results(result_id: str):
    # Only accept IDs generated by this API.
    if (
        len(result_id) != 32
        or any(
            char not in "0123456789abcdef"
            for char in result_id
        )
    ):
        raise HTTPException(
            status_code=404,
            detail="Result not found.",
        )

    output_path = (
        RESULTS_DIR / f"{result_id}_predictions.csv"
    )

    if not output_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Result not found.",
        )

    return FileResponse(
        path=output_path,
        media_type="text/csv",
        filename="directionlab_predictions.csv",
    )
