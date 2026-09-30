# DirectionLab — AI-Based Cyber Threat Detection

**SIH 2026 — Problem Statement ID: 26145**

> **AI-Based Detection of Cyber Threats in Unidirectional IP Traffic**

DirectionLab is an AI-powered cybersecurity application designed to detect potential cyber threats in **unidirectional IP network-flow traffic**.

The system accepts a network-flow CSV file, processes the traffic using a trained machine-learning model, and produces predictions identifying traffic as **benign** or **potentially malicious/anomalous**.

The project provides a lightweight web interface using a static HTML frontend and a **FastAPI** backend for model inference.

---

## Features

* Upload network-flow CSV files
* Drag-and-drop CSV upload
* AI/ML-based traffic analysis
* Binary threat prediction
* Benign vs. attack/anomaly classification
* Prediction statistics
* Attack/anomaly rate calculation
* Configurable prediction threshold
* Downloadable analysis results
* REST API for inference
* Health-check endpoint
* Research-oriented detection workflow

> **Important:** DirectionLab provides a research/AI detection signal. A prediction should not be treated as definitive proof of a cyberattack without further investigation.

---

# 1. Project Architecture

The project consists of two main components:

```text
DirectionLab
│
├── Frontend
│   └── src/
│       └── directionlab/
│           └── static/
│               └── index.html
│
└── Backend
    ├── directionlab/
    │   ├── api.py
    │   └── ...
    │
    ├── model/checkpoint files
    ├── pyproject.toml
    └── ...
```

### Frontend

The frontend is a **static HTML application**.

Main file:

```text
src/directionlab/static/index.html
```

It provides:

* Dataset upload
* Drag-and-drop support
* Analysis controls
* Results display
* Prediction statistics
* Download results

### Backend

The backend is built using **FastAPI**.

The API is responsible for:

1. Receiving the uploaded CSV
2. Validating/processing the data
3. Loading the trained model
4. Running inference
5. Generating predictions
6. Returning analysis statistics
7. Providing downloadable results

The FastAPI application is exposed through:

```text
directionlab.api:app
```

---

# 2. Requirements

Before running DirectionLab, install the following.

## Required Software

### Python

Install **Python 3.10 or newer**.

Check your Python installation:

```powershell
python --version
```

or:

```powershell
py --version
```

The project uses Python for the FastAPI backend and machine-learning inference.

---

## Git

Git is recommended if you are cloning the repository.

Check:

```powershell
git --version
```

If Git is not installed, install it before cloning the project.

---

# 3. Python Dependencies

The backend requires Python packages for the API, file uploads, model execution, data processing, and supporting functionality.

The important runtime dependencies include:

```text
fastapi
uvicorn
python-multipart
```

FastAPI requires `python-multipart` when receiving uploaded files through multipart form data.

Additional project dependencies are defined by the project's Python package configuration.

If the repository contains a `pyproject.toml`, install the project dependencies from the project root rather than manually installing every package.

---

# 4. Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/Varad-Nagari16/directionlab-cyber-threat-detection.git
```

Move into the repository:

```powershell
cd directionlab-cyber-threat-detection
```

The project root should contain the Python project files and the `src` directory.

---

# 5. Create a Virtual Environment

It is recommended to use a virtual environment so that DirectionLab's Python dependencies remain isolated from other Python projects.

Run:

```powershell
python -m venv .venv
```

If `python` does not work, use:

```powershell
py -m venv .venv
```

---

# 6. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, your terminal should look similar to:

```text
(.venv) PS C:\...\directionlab-cyber-threat-detection>
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# 7. Install Dependencies

First upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Then install the project:

```powershell
python -m pip install -e .
```

This installs the project and its declared Python dependencies.

If the project environment does not automatically include the API dependencies, install the required runtime packages:

```powershell
python -m pip install fastapi uvicorn python-multipart
```

For file-upload functionality, `python-multipart` is required by FastAPI.

---

# 8. Verify the Installation

Check FastAPI:

```powershell
python -c "import fastapi; print(fastapi.__version__)"
```

Check Uvicorn:

```powershell
python -m uvicorn --version
```

Check that the project can be imported:

```powershell
python -c "from directionlab.api import app; print('DirectionLab backend loaded successfully')"
```

If these commands complete without an error, the backend environment is ready.

---

# 9. Start the Backend

Make sure you are in the **project root** and the virtual environment is activated.

Run:

```powershell
python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8000
```

You should see something similar to:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Keep this PowerShell window open.

### Do not close the backend terminal

The FastAPI server must remain running while using the application.

---

# 10. Check Whether the Backend Is Running

Open your browser and visit:

```text
http://localhost:8000/health
```

A successful response should look similar to:

```json
{
  "status": "ok",
  "checkpoint_exists": true,
  "frontend_exists": true
}
```

If you receive a successful response, the DirectionLab backend is running correctly.

---

# 11. Open the DirectionLab Frontend

The main frontend file is:

```text
src/directionlab/static/index.html
```

### Recommended way

Because the HTML frontend communicates with the FastAPI backend, run the backend first and access the frontend through the application.

If your FastAPI application exposes the frontend, open:

```text
http://localhost:8000/app
```

This keeps the frontend and backend on the same server.

---

## Alternative: Open the HTML File Directly

You can also locate:

```text
src\directionlab\static\index.html
```

and open it in your browser.

For example:

```text
directionlab-cyber-threat-detection
└── src
    └── directionlab
        └── static
            └── index.html
```

### Important

Opening the HTML file directly is useful for viewing the interface, but **AI inference still requires the FastAPI backend to be running**.

The frontend sends the uploaded dataset to the backend's `/predict` endpoint.

Therefore, for the complete working application:

```text
Frontend
   ↓
index.html
   ↓
Upload CSV
   ↓
FastAPI /predict
   ↓
ML Model
   ↓
Predictions
   ↓
Results displayed in frontend
```

---

# 12. Complete Run Walkthrough

This is the easiest way to run the project from a fresh clone.

### Step 1 — Clone

```powershell
git clone https://github.com/Varad-Nagari16/directionlab-cyber-threat-detection.git
```

### Step 2 — Enter project

```powershell
cd directionlab-cyber-threat-detection
```

### Step 3 — Create virtual environment

```powershell
python -m venv .venv
```

### Step 4 — Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5 — Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

If necessary:

```powershell
python -m pip install fastapi uvicorn python-multipart
```

### Step 6 — Start backend

```powershell
python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8000
```

### Step 7 — Verify backend

Open:

```text
http://localhost:8000/health
```

### Step 8 — Open frontend

Open:

```text
src/directionlab/static/index.html
```

or, if the application serves the static frontend:

```text
http://localhost:8000/app
```

### Step 9 — Upload dataset

On the DirectionLab interface:

1. Go to **Analyze a Dataset**
2. Select a `.csv` network-flow dataset
3. Wait for the upload
4. The frontend sends the file to `/predict`
5. The backend processes the dataset
6. The trained model performs inference
7. Results are displayed on the dashboard

---

# 13. Using the Dataset Analyzer

The main workflow is:

```text
CSV Dataset
     │
     ▼
Upload
     │
     ▼
FastAPI Backend
     │
     ▼
Data Processing
     │
     ▼
Trained Model
     │
     ▼
Prediction
     │
     ├── Benign
     │
     └── Attack / Anomaly
     │
     ▼
Result Statistics
     │
     ▼
Download Results
```

The dashboard displays information such as:

* Number of rows analyzed
* Number of attack/anomaly predictions
* Number of benign predictions
* Detection/attack rate
* Prediction threshold
* Distribution of predictions
* Downloadable results

---

# 14. CSV Dataset

The frontend accepts:

```text
.csv
```

files.

The dataset should contain the network-flow features expected by the trained DirectionLab model.

### Important

A random CSV file will not necessarily work.

The uploaded dataset must contain compatible network-flow features expected by the model and preprocessing pipeline.

For best results, use the dataset format for which the model was trained/evaluated.

---

# 15. API Endpoints

The backend exposes API endpoints for application functionality.

## Health Check

```http
GET /health
```

Used to verify that the backend and required model resources are available.

Example:

```text
http://localhost:8000/health
```

---

## Prediction

```http
POST /predict
```

Accepts the uploaded CSV dataset and performs model inference.

The frontend sends the uploaded file to this endpoint.

The request uses multipart form data because a file is being uploaded. FastAPI supports file uploads through `File`/`UploadFile` and multipart form data.

---

## Results

```http
GET /results/{result_id}
```

Used to retrieve generated analysis results associated with a result ID.

---

# 16. Frontend Technologies

The frontend is intentionally lightweight.

### Technologies

* HTML5
* CSS3
* JavaScript
* Fetch API
* HTML drag-and-drop API
* FormData API

No Node.js installation is required just to run the static frontend.

The primary frontend file is:

```text
src/directionlab/static/index.html
```

---

# 17. Backend Technologies

### Python

The backend and ML pipeline are implemented in Python.

### FastAPI

Used to provide the REST API and handle dataset uploads.

### Uvicorn

Used as the ASGI server to run the FastAPI application. Uvicorn can be installed through pip and is commonly used to serve FastAPI applications.

### Machine Learning Model

The backend loads the trained DirectionLab model/checkpoint and performs inference against uploaded network-flow data.

---

# 18. Troubleshooting

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'fastapi'
```

Make sure the virtual environment is activated:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```powershell
python -m pip install -e .
```

If necessary:

```powershell
python -m pip install fastapi uvicorn python-multipart
```

---

## `python-multipart` Error

If you see an error related to file/form uploads, install:

```powershell
python -m pip install python-multipart
```

FastAPI requires this package to process uploaded files sent as form data.

---

## `Failed to fetch`

If the frontend displays:

```text
Failed to fetch
```

check that the FastAPI backend is running.

Start it with:

```powershell
python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8000
```

Then verify:

```text
http://localhost:8000/health
```

If `/health` does not load, the backend is not running correctly.

---

## Port 8000 Already in Use

If port `8000` is already being used, stop the other process or run DirectionLab on another port:

```powershell
python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8001
```

Then use:

```text
http://localhost:8001
```

---

## PowerShell Does Not Allow `.venv` Activation

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Model/Checkpoint Error

If the health endpoint reports:

```json
"checkpoint_exists": false
```

the required trained model checkpoint is not available at the expected project location.

Check that the model/checkpoint files included with the project have not been moved or deleted.

---

# 19. Recommended Development Workflow

When developing DirectionLab, use two terminal windows.

### Terminal 1 — Backend

```powershell
cd directionlab-cyber-threat-detection
.\.venv\Scripts\Activate.ps1

python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 — Development / Git

Use the second terminal for:

```powershell
git status
```

and other development commands.

The `--reload` option automatically reloads the FastAPI server when backend Python files change.

---

# 20. Project Structure

A simplified project structure is:

```text
directionlab-cyber-threat-detection/
│
├── src/
│   └── directionlab/
│       │
│       ├── static/
│       │   └── index.html
│       │
│       ├── api.py
│       └── ...
│
├── .venv/
│
├── pyproject.toml
│
├── README.md
│
└── model/checkpoint files
```

> The `.venv` directory is created locally and should not normally be committed to Git.

---

# 21. One-Command Backend Start

After the environment has already been created and dependencies installed, the backend can be started with:

```powershell
python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8000
```

Then open the DirectionLab frontend.

---

# 22. Quick Start

For someone who already has Python installed:

```powershell
git clone https://github.com/Varad-Nagari16/directionlab-cyber-threat-detection.git

cd directionlab-cyber-threat-detection

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -e .

python -m pip install fastapi uvicorn python-multipart

python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/health
```

and open the DirectionLab frontend:

```text
src/directionlab/static/index.html
```

---

# 23. Research Scope

DirectionLab focuses on detecting cyber threats using **network-flow metadata rather than requiring direct payload inspection**.

The broader research direction includes:

* Flow-based intrusion detection
* Temporal/cause-aware traffic analysis
* Direction-preserving feature handling
* Cross-dataset evaluation
* Universal model development
* Binary and multiclass threat detection

The model's predictions should be interpreted as detection signals that require appropriate cybersecurity investigation and validation.

---

# 24. Team / Project Information

**Project:** DirectionLab

**SIH Problem Statement:** 26145

**Organization:** National Technical Research Organisation (NTRO)

**Category:** Software

**Theme:** Blockchain & Cybersecurity

**Project Type:** AI/ML-based Cybersecurity

---

# 25. License

Refer to the repository's license file for the applicable project licensing terms.

---

## DirectionLab

**AI-Based Detection of Cyber Threats in Unidirectional IP Traffic**

Built for **Smart India Hackathon 2026**.
