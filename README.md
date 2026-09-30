# MLOps Model Serving Platform

Production-style machine learning model serving platform demonstrating an end-to-end MLOps workflow using **PyTorch, MLflow, FastAPI, Docker, CI/CD, and Azure**.

The project covers the lifecycle of a machine learning model from training and experiment tracking through model registration, versioning, deployment, and API-based inference.

---

## Overview

This project demonstrates how a trained machine learning model can be moved from a development environment into a reproducible, containerized serving environment.

The platform implements the following workflow:

```text
                    ┌─────────────────────┐
                    │     PyTorch Model   │
                    │      Training       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       MLflow        │
                    │ Experiment Tracking │
                    │  Metrics & Params   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Registry    │
                    │ Versions + Aliases  │
                    │     "champion"      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │   Model Loading &   │
                    │      Inference      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Docker        │
                    │ Containerized API   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   CI/CD + Azure     │
                    │ Automated Delivery  │
                    └─────────────────────┘
```

---

## What This Project Demonstrates

The project focuses on practical MLOps engineering patterns:

* PyTorch model development and training
* Deterministic dataset generation
* MLflow experiment tracking
* Hyperparameter and metric logging
* MLflow model artifact management
* Model registration and versioning
* MLflow model aliases
* Dynamic model loading through a registry
* FastAPI model serving
* Request/response validation with Pydantic
* Docker containerization
* Docker Compose orchestration
* Automated testing
* CI/CD with GitHub Actions
* Infrastructure-as-code using Azure Bicep
* Production-oriented configuration using environment variables

---

## Technology Stack

| Area                 | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Machine Learning     | PyTorch               |
| Experiment Tracking  | MLflow                |
| Model Registry       | MLflow Model Registry |
| API                  | FastAPI               |
| Validation           | Pydantic              |
| Containerization     | Docker                |
| Local Orchestration  | Docker Compose        |
| Testing              | Pytest                |
| Code Quality         | Ruff                  |
| CI/CD                | GitHub Actions        |
| Cloud                | Microsoft Azure       |
| Infrastructure       | Azure Bicep           |
| Version Control      | Git / GitHub          |

---

## Project Structure

```text
mlops-model-serving-platform/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy-azure.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
│
├── ml/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── infra/
│   └── azure-container-apps.bicep
│
├── .dockerignore
├── .env.example
├── .gitignore
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

The dataset contains four numerical features:

```text
Feature 1
Feature 2
Feature 3
Feature 4
```

A deterministic random seed is used so that training runs are reproducible.

The generated data is split into:

```text
80% Training
20% Validation
```

---

## 2. PyTorch Model

The classifier is implemented using PyTorch.

Architecture:

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

Training uses:

* Adam optimizer
* Binary Cross Entropy loss
* 30 training epochs
* Learning rate of `0.01`

---

# MLflow Experiment Tracking

MLflow is used to track training experiments and manage model artifacts.

The training pipeline records:

### Parameters

```text
epochs
learning_rate
architecture
optimizer
loss_function
```

### Metrics

```text
validation loss
validation accuracy
```

### Model

The trained PyTorch model is logged to MLflow with:

* Model signature
* Example input
* Serialized model artifact
* Model metadata

---

# Model Registry

The trained model is registered in the MLflow Model Registry.

Example model:

```text
ticket_classifier
```

Model versions allow multiple trained models to coexist.

For example:

```text
ticket_classifier
│
├── Version 1
├── Version 2
└── Version 3
```

The API does not need to hard-code a specific model version.

Instead, it loads the model using an MLflow alias:

```text
models:/ticket_classifier@champion
```

This allows the deployed API to consume whichever model version is assigned to the `champion` alias.

This provides a simple mechanism for model promotion without changing application code.

---

# Model Serving

The model is served through FastAPI.

The API loads the registered MLflow model when inference is requested.

The serving flow is:

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
Champion Model
     │
     ▼
Prediction
     │
     ▼
JSON Response
```

---

# API Endpoints

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

---

## Model Information

```http
GET /model
```

Example response:

```json
{
  "model_name": "ticket_classifier",
  "model_alias": "champion",
  "tracking_uri": "http://mlflow:5000"
}
```

---

## Prediction

```http
POST /predict
```

Example request:

```json
{
  "features": [
    0.1,
    0.2,
    0.3,
    0.4
  ]
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.5296782851219177,
  "model_name": "ticket_classifier",
  "model_alias": "champion"
}
```

The API validates that exactly four numerical features are supplied.

---

# Interactive API Documentation

Once the API is running, FastAPI provides interactive documentation.

Open:

```text
http://localhost:8000/docs
```

This can be used to test:

* Health checks
* Model metadata
* Prediction requests

---

# Local Development

## Prerequisites

Install:

* Python 3.14+
* Docker Desktop
* Git

Verify Python:

```powershell
python --version
```

Verify Docker:

```powershell
docker --version
docker compose version
```

---

## Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# Running MLflow and FastAPI

The recommended local setup uses Docker Compose.

Start the services:

```powershell
docker compose up --build
```

This starts:

```text
FastAPI
    │
    └── http://localhost:8000

MLflow
    │
    └── http://localhost:5000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

MLflow UI:

```text
http://localhost:5000
```

---

# Training the Model

The training script connects to the MLflow tracking server and creates a training run.

From the project environment:

```powershell
python -m ml.train
```

The training pipeline:

1. Creates the dataset
2. Builds the PyTorch model
3. Trains the model
4. Evaluates validation performance
5. Logs parameters to MLflow
6. Logs metrics to MLflow
7. Creates a model signature
8. Logs the model artifact
9. Registers the model

---

# Model Promotion

After training, MLflow creates a new model version.

The model can then be assigned an alias such as:

```text
champion
```

The serving application resolves:

```text
models:/ticket_classifier@champion
```

rather than depending on a hard-coded model version.

This separates:

```text
Model training
```

from:

```text
Model deployment
```

and makes model promotion easier to manage.

---

# Docker

The API is packaged into a Docker image.

Build the API image:

```powershell
docker compose build api
```

Start the application:

```powershell
docker compose up
```

Check running containers:

```powershell
docker compose ps
```

View API logs:

```powershell
docker compose logs api
```

View MLflow logs:

```powershell
docker compose logs mlflow
```

---

# Testing

The project includes automated API tests using Pytest.

Run tests locally:

```powershell
pytest
```

For a clean environment using the same containerized services:

```powershell
docker compose up --build
```

The test suite is designed to validate API behaviour without coupling production images to development-only test dependencies.

---

# Code Quality

Ruff is used for linting and code quality checks.

Run Ruff:

```powershell
ruff check .
```

The project also uses `pyproject.toml` to centralize Python tooling configuration.

---

# CI/CD

GitHub Actions is used to automate validation and deployment workflows.

The CI workflow performs automated checks such as:

```text
Push / Pull Request
        │
        ▼
Install dependencies
        │
        ▼
Run linting
        │
        ▼
Run tests
        │
        ▼
Build application
```

This provides an automated quality gate before changes are merged or deployed.

---

# Azure Deployment

The project includes Azure infrastructure configuration using Bicep.

Infrastructure is defined in:

```text
infra/azure-container-apps.bicep
```

The intended deployment architecture is:

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

This demonstrates infrastructure-as-code and cloud deployment practices alongside the machine learning workflow.

---

# Configuration

Application configuration is controlled through environment variables.

Example:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_MODEL_NAME=ticket_classifier
MLFLOW_MODEL_ALIAS=champion
```

The project includes:

```text
.env.example
```

Sensitive or machine-specific configuration should not be committed to Git.

---

# Engineering Decisions

## Model Registry Instead of Bundling Models

The application does not package a specific trained model directly inside the API image.

Instead, the API retrieves the configured model from MLflow.

This keeps:

```text
Application code
```

separate from:

```text
Model artifacts
```

---

## Alias-Based Model Loading

The API uses:

```text
models:/ticket_classifier@champion
```

instead of:

```text
models:/ticket_classifier/3
```

This allows model versions to change independently from the API deployment.

---

## Containerized MLflow

MLflow runs as a separate service from the FastAPI application.

This provides a clearer separation between:

```text
ML platform services
```

and:

```text
Inference services
```

---

## Production Image Excludes Tests

The production API image contains the application and model-serving dependencies but does not include the test suite.

Testing dependencies and production dependencies can therefore remain separated.

---

# Reproducibility

The project uses several practices to improve reproducibility:

* Deterministic dataset generation
* Explicit dependency management
* Docker containerization
* Environment-based configuration
* MLflow experiment tracking
* Model versioning
* Model signatures
* Infrastructure-as-code
* Automated CI checks

---

# Future Improvements

Potential next steps include:

* Real-world ticket classification dataset
* Data versioning
* Feature store integration
* Model performance monitoring
* Drift detection
* Prometheus metrics
* Grafana dashboards
* Structured logging
* Distributed training
* GPU-based training
* Automated model evaluation gates
* Automated model promotion
* Canary deployments
* Blue/green deployments
* Azure ML integration
* Kubernetes deployment
* Security and authentication for the inference API

---


