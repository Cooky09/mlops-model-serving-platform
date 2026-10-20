# MLOps Model Serving Platform

Production-style machine learning model serving platform demonstrating an end-to-end MLOps workflow using **PyTorch, MLflow, FastAPI, Docker, GitHub Actions, and Microsoft Azure**.

The project demonstrates the lifecycle of a machine learning model from deterministic training and experiment tracking through model registration, versioning, promotion, rollback, and API-based inference.

---

## Overview

The platform demonstrates how a trained machine learning model can move from development into a reproducible, containerized serving environment.

### Architecture

```text
┌─────────────────────┐
│   PyTorch Model     │
│      Training       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       MLflow        │
│ Experiment Tracking │
│   Metrics & Params  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Model Registry    │
│ Versions + Aliases  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Model Promotion /   │
│      Rollback       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     FastAPI API     │
│ Model Loading &     │
│     Inference       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       Docker        │
│   Containerized API │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CI/CD + Azure     │
│ Automated Delivery  │
└─────────────────────┘
```

## What This Project Demonstrates

- PyTorch model development and training
- Deterministic dataset generation
- Reproducible training configuration
- MLflow experiment tracking
- Hyperparameter and metric logging
- MLflow model artifact management
- Model signatures
- Model registration and versioning
- MLflow model aliases
- Model promotion and rollback
- Model lifecycle testing
- Dynamic model loading through a registry
- FastAPI model serving
- Pydantic request/response validation
- Docker containerization
- Docker Compose orchestration
- Automated testing with Pytest
- Code quality checks with Ruff
- CI/CD with GitHub Actions
- Infrastructure-as-code with Azure Bicep
- Production-oriented environment configuration

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Machine Learning | PyTorch |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API | FastAPI |
| Validation | Pydantic |
| Containerization | Docker |
| Local Orchestration | Docker Compose |
| Testing | Pytest |
| Code Quality | Ruff |
| CI/CD | GitHub Actions |
| Cloud | Microsoft Azure |
| Infrastructure | Azure Bicep |
| Version Control | Git / GitHub |

---

## Project Structure

```text
mlops-model-serving-platform/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── deploy-azure.yml
│   └── pull_request_template.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── ml/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── lifecycle.py
│   ├── model.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_lifecycle.py
│   ├── test_model.py
│   └── test_model_persistence.py
├── infra/
│   └── azure-container-apps.bicep
├── .dockerignore
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── Dockerfile
├── Dockerfile.mlflow
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Machine Learning Workflow

## 1. Dataset Generation

The project generates a deterministic binary classification dataset using NumPy.

The dataset contains four numerical features.

The current configuration uses:

```text
Random seed: 42
Samples:     1000
Features:    4
```

The generated data is split into:

- 80% training data
- 20% validation data

Implementation:

```text
ml/data.py
```

## 2. PyTorch Model

The classifier is implemented using PyTorch.

```text
Input: 4 features
    │
    ▼
Linear(4 → 16)
    │
   ReLU
    │
    ▼
Linear(16 → 8)
    │
   ReLU
    │
    ▼
Linear(8 → 1)
    │
  Sigmoid
    │
    ▼
Binary Prediction
```

Current training configuration:

```text
Epochs:        30
Learning rate: 0.01
Optimizer:     Adam
```

Implementation:

```text
ml/model.py
```

## Model Evaluation

Evaluation is separated from training.

The evaluation pipeline calculates:

- Accuracy
- Precision
- Recall
- F1 Score

Implementation:

```text
ml/evaluate.py
```

---

# MLflow

## Experiment Tracking

MLflow tracks training experiments and model artifacts.

Tracked parameters include:

- `epochs`
- `learning_rate`
- `random_seed`
- `n_samples`

Tracked metrics include:

- `accuracy`
- `precision`
- `recall`
- `f1`

The trained model is logged with:

- Model signature
- Example input
- Serialized model artifact
- Model metadata
- Registered model name

Implementation:

```text
ml/train.py
```

## Model Registry

The trained model is registered in the MLflow Model Registry.

Default model:

```text
ticket_classifier
```

Multiple versions can coexist:

```text
ticket_classifier
├── Version 1
├── Version 2
└── Version 3
```

Each version represents a distinct model artifact.

---

# Model Lifecycle

The platform implements a complete model lifecycle:

```text
Training
   │
   ▼
Model Registration
   │
   ▼
New Model Version
   │
   ▼
Production Alias
   │
   ▼
Model Serving
   │
   ├──────────────┐
   │              │
   │              ▼
   │           Rollback
   │              │
   │              ▼
   └──────► Previous Version
```

### Production Alias

The default production alias is:

```text
production
```

The API resolves the model using:

```text
models:/ticket_classifier@production
```

This allows the model version behind the alias to change without modifying application code.

### Rollback

Rollback is performed by moving the `production` alias to a previous registered model version.

Implementation:

```text
ml/lifecycle.py
```

## Lifecycle Testing

Lifecycle tests verify:

- Production alias movement
- Model promotion
- Model rollback
- MLflow client interaction
- Lifecycle behaviour without a live MLflow deployment

Implementation:

```text
tests/test_lifecycle.py
```

---

# Model Serving

The model is served through FastAPI.

```text
HTTP Request
     │
     ▼
FastAPI
     │
     ▼
Pydantic Validation
     │
     ▼
Model Loader
     │
     ▼
MLflow Model Registry
     │
     ▼
Production Model
     │
     ▼
Prediction
     │
     ▼
JSON Response
```

## API Endpoints

### `GET /health`

Returns the application health status.

```json
{
  "status": "healthy"
}
```

### `GET /model`

Returns the configured model metadata.

```json
{
  "model_name": "ticket_classifier",
  "model_alias": "production",
  "tracking_uri": "http://mlflow:5000"
}
```

### `POST /predict`

Example request:

```json
{
  "features": [0.1, 0.2, 0.3, 0.4]
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.5296782851219177,
  "model_name": "ticket_classifier",
  "model_alias": "production"
}
```

The API validates that exactly four numerical features are supplied.

## Interactive API Documentation

When the API is running:

```text
http://localhost:8000/docs
```

---

# Local Development

## Prerequisites

- Python 3.14+
- Docker Desktop
- Git

Verify the installations:

```bash
python --version
docker --version
docker compose version
```

## Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Platform

The recommended local setup uses Docker Compose:

```bash
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| FastAPI | `http://localhost:8000` |
| FastAPI Docs | `http://localhost:8000/docs` |
| MLflow | `http://localhost:5000` |

---

# Training

Run:

```bash
python -m ml.train
```

The training pipeline:

1. Creates the deterministic dataset
2. Builds the PyTorch model
3. Trains the model
4. Evaluates validation performance
5. Calculates classification metrics
6. Logs parameters to MLflow
7. Logs metrics to MLflow
8. Creates a model signature
9. Logs the model artifact
10. Registers the model
11. Assigns the configured alias

---

# Docker

Build the API:

```bash
docker compose build api
```

Start the services:

```bash
docker compose up
```

Check containers:

```bash
docker compose ps
```

View API logs:

```bash
docker compose logs api
```

View MLflow logs:

```bash
docker compose logs mlflow
```

---

# Testing

The project includes automated tests covering:

- API behaviour
- Dataset generation
- Evaluation metrics
- Model structure
- Model persistence
- Model lifecycle behaviour
- Drift detection and reporting

Run the test suite:

```bash
pytest
```

The tests are designed to avoid unnecessary dependencies on a live MLflow service.

---

# Code Quality

Ruff is used for linting.

Run:

```bash
python -m ruff check .
```

Configuration is maintained in:

```text
pyproject.toml
```

---

# CI/CD

GitHub Actions automates validation and deployment.

```text
Pull Request / Push
        │
        ▼
Install dependencies
        │
        ▼
Run Ruff
        │
        ▼
Run Pytest
        │
        ▼
Build / validation
```

Deployment automation is maintained separately from CI.

---

# Azure Deployment

The production deployment uses Azure Container Apps and Azure Container Registry.

Infrastructure is defined with:

```text
infra/azure-container-apps.bicep
```

The deployment architecture is:

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ▼
Container Build
   │
   ▼
Azure Container Registry
   │
   ▼
Azure Container Apps
   │
   ▼
FastAPI Model Serving
```

Production MLflow configuration is supplied through environment variables rather than being hard-coded into the application.

---

# Configuration

Application configuration is controlled through environment variables.

Example:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_MODEL_NAME=ticket_classifier
MLFLOW_MODEL_ALIAS=production
```

The repository includes:

```text
.env.example
```

Sensitive or machine-specific configuration must not be committed to Git.

---

# Engineering Decisions

## Model Registry Instead of Bundling Models

The API does not package a specific trained model directly into the application image.

Instead, it retrieves the configured model from MLflow.

This separates:

```text
Application Code
       │
       └── Model Artifacts
```

and allows models to be updated independently of application code.

## Alias-Based Model Loading

The API uses:

```text
models:/ticket_classifier@production
```

rather than:

```text
models:/ticket_classifier/3
```

This allows the production model version to change without modifying the application.

## Registry-Based Rollback

Rollback is performed by moving the production alias to a previous registered version.

No API image rebuild is required.

## Deterministic Training

Training uses explicit configuration and a fixed random seed to improve reproducibility.

## Containerized Services

MLflow and FastAPI are separated into services for local development, creating a clear boundary between platform services and inference services.

---

# Reproducibility

The project uses:

- Deterministic dataset generation
- Explicit random seed
- Explicit dependency management
- Docker containerization
- Environment-based configuration
- MLflow experiment tracking
- Model versioning
- Model signatures
- Infrastructure-as-code
- Automated CI checks
- Automated test coverage

---

# Project Status

The project is developed incrementally through GitHub Issues and pull requests.

## Phase 1 — Engineering Foundation

**Complete**

- GitHub development workflow
- Pull request templates
- Issue templates
- Contributor workflow
- Branch protection
- CI quality gates

## Phase 2 — ML Pipeline

**Complete**

- Deterministic dataset generation
- Reproducible training configuration
- Model evaluation metrics
- Model artifact validation
- Model persistence validation

## Phase 3 — Model Lifecycle

**Complete**

- MLflow model registration
- Model version management
- Model promotion
- Production alias management
- Model rollback
- Lifecycle integration testing

## Phase 4 — Observability

**Complete**

- Structured application logging
- API request metrics
- Request error metrics
- Observability integration and validation

## Phase 5 — Data / Model Drift

**Complete**

- Data drift detection
- Prediction drift detection
- Drift reporting
- Drift monitoring documentation

## Phase 6 — Deployment Automation

**Complete**

- Production API containerization
- Production MLflow configuration
- Azure production deployment
- Deployment health checks

## Phase 7 — Production Hardening

**Next**

The next planned phase focuses on securing and hardening the production platform.

Planned areas:

- API authentication
- Secrets management
- Network security
- Rate limiting
- Dependency vulnerability scanning
- Model access controls

---

# Drift Monitoring

The platform includes automated checks for both input data drift and prediction drift.

## Data Drift

Data drift compares a baseline feature distribution with a current feature distribution.

The current implementation compares feature means and flags drift when the difference exceeds a configurable threshold.

```python
calculate_feature_drift(
    baseline,
    current,
    threshold=0.1,
)
```

## Prediction Drift

Prediction drift compares the mean of baseline predictions with the mean of current predictions.

```python
calculate_prediction_drift(
    baseline,
    current,
    threshold=0.1,
)
```

## Drift Reporting

The two checks can be combined into a machine-readable report:

```python
build_drift_report(
    data_drift=True,
    prediction_drift=False,
)
```

The report contains:

- `data_drift`
- `prediction_drift`
- `drift_detected`

This provides a foundation for future monitoring and deployment workflows.

---

# Engineering Workflow

Development follows this workflow:

```text
Issue
  ↓
Feature Branch
  ↓
Implementation
  ↓
Tests
  ↓
Ruff
  ↓
Commit
  ↓
Pull Request
  ↓
CI
  ↓
Merge
```

The repository uses:

- GitHub Issues for work tracking
- Feature branches
- Pull requests
- GitHub Actions
- Protected `main`
- Required CI checks
- Issue-linked pull requests

Contributor documentation:

```text
CONTRIBUTING.md
```

---

# Project Goal

The long-term goal is to demonstrate a complete production-oriented MLOps platform:

```text
Data
 │
 ▼
Training
 │
 ▼
Evaluation
 │
 ▼
Experiment Tracking
 │
 ▼
Model Registration
 │
 ▼
Version Management
 │
 ▼
Promotion
 │
 ▼
Serving
 │
 ▼
Observability
 │
 ▼
Drift Detection
 │
 ▼
Deployment
 │
 ▼
Production Hardening
```

The emphasis is on **reproducibility, automation, testability, model lifecycle management, observability, and production-oriented engineering practices** rather than simply training a machine learning model.
