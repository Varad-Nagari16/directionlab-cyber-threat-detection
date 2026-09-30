# DirectionLab

## AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

DirectionLab is an AI-based cyber threat detection system designed to analyze network-flow data and identify potential malicious traffic using flow-level features, directional information, and temporal context.

The project is developed for **Smart India Hackathon 2026 – Problem Statement 26145**, under the **Blockchain & Cybersecurity** theme for the **National Technical Research Organisation (NTRO)**.

---

## Project Overview

DirectionLab focuses on detecting cyber threats from network-flow metadata without requiring inspection or decryption of application payloads.

The system is designed around:

* Passive / read-only traffic analysis
* Network-flow metadata
* Direction-aware analysis
* Temporal context
* AI/ML-based threat detection
* Support for compatible network-flow datasets
* Attack and benign traffic classification
* Downloadable prediction results

The dashboard provides a simple interface where a user can upload a supported CSV dataset and run the DirectionLab model.

---

## Project Structure

The important part of the project for running the current interface is:

```text
directionlab-cyber-threat-detection/
│
├── src/
│   └── directionlab/
│       └── static/
│           └── index.html
│
├── ...
└── README.md
```

The current frontend/dashboard is implemented as a **static HTML file**:

```text
src/directionlab/static/index.html
```

You do **not** need to install Node.js, Next.js, Vite, React, or any frontend framework just to view the current dashboard.

---

# How to Run

## Option 1 — Open the dashboard directly

The simplest way to view the current UI is to open:

```text
src/directionlab/static/index.html
```

### Steps

1. Clone the repository:

```bash
git clone https://github.com/Varad-Nagari16/directionlab-cyber-threat-detection.git
```

2. Enter the project directory:

```bash
cd directionlab-cyber-threat-detection
```

3. Open:

```text
src/directionlab/static/index.html
```

in a web browser.

That's it for viewing the static dashboard.

---

# Running with the Backend

The `index.html` file is the frontend interface for the DirectionLab inference system.

If you want the **CSV upload and AI prediction functionality** to work, the DirectionLab backend must also be running.

The backend is implemented using FastAPI.

## 1. Create a virtual environment

From the project root:

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install the project

From the project root:

```bash
python -m pip install -e .
```

If FastAPI/Uvicorn are not already installed:

```bash
python -m pip install fastapi uvicorn python-multipart
```

---

## 3. Start the backend

Run:

```bash
python -m uvicorn directionlab.api:app --reload --host 0.0.0.0 --port 8000
```

The backend should start at:

```text
http://localhost:8000
```

You can verify it using:

```text
http://localhost:8000/health
```

A healthy backend should return a response similar to:

```json
{
  "status": "ok",
  "checkpoint_exists": true,
  "frontend_exists": true
}
```

---

# Using the Dashboard

Once the backend is running:

1. Open the DirectionLab dashboard.
2. Select a supported network-flow CSV.
3. Drop the CSV into the upload area or click **Choose CSV file**.
4. Click **Run analysis**.
5. The dataset is sent to the `/predict` API endpoint.
6. The backend performs model inference.
7. The dashboard displays:

   * Flows analyzed
   * Attack predictions
   * Benign predictions
   * Predicted attack rate
8. The generated prediction file can then be downloaded.

---

# API Endpoints

The current backend exposes the following main endpoints:

| Endpoint               | Method | Purpose                               |
| ---------------------- | ------ | ------------------------------------- |
| `/health`              | GET    | Check backend and model status        |
| `/predict`             | POST   | Upload a CSV and run inference        |
| `/results/{result_id}` | GET    | Retrieve generated prediction results |

---

# Dataset

The dashboard expects a **CSV containing supported network-flow data**.

The backend performs dataset/schema detection and validation before running inference.

The uploaded data should contain the required flow-level features expected by the DirectionLab model.

---

# Threat Detection Scope

The project focuses on network-level indicators associated with threat categories such as:

* DDoS
* Botnet command-and-control beaconing
* DGA / DNS tunnelling
* Malware activity over encrypted sessions
* Reconnaissance and port scanning
* Data exfiltration

The system operates on network-flow metadata rather than requiring application payload decryption.

---

# Technology Stack

### Frontend

* HTML
* CSS
* JavaScript
* Static dashboard

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* PyTorch
* scikit-learn
* NumPy
* Pandas

---

# Important Note

The current dashboard is intentionally implemented as a **static `index.html` frontend**.

You do not need a frontend development server to view the interface.

The frontend can be opened directly from:

```text
src/directionlab/static/index.html
```

However, **AI inference requires the FastAPI backend to be running**, because the dashboard sends uploaded datasets to the backend `/predict` endpoint.

---

# Development

For frontend UI changes, edit:

```text
src/directionlab/static/index.html
```

The file contains the dashboard's:

* HTML structure
* CSS styling
* JavaScript functionality
* CSV upload interface
* Backend API interaction
* Results display

After modifying the file, simply refresh the browser to see the changes.

---

# Project Status

DirectionLab currently provides a working research dashboard for uploading compatible network-flow datasets and obtaining model-backed predictions through the FastAPI inference API.

The architecture is designed to support further development toward real-time or near-real-time network threat detection.

---

## Problem Statement

**SIH 2026 – Problem Statement 26145**

**AI-Based Detection of Cyber Threats in Unidirectional IP Traffic**

**Organization:** National Technical Research Organisation (NTRO)

**Theme:** Blockchain & Cybersecurity

---

## Repository

GitHub:

https://github.com/Varad-Nagari16/directionlab-cyber-threat-detection
