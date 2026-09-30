# MLOps Model Serving Platform

A portfolio-grade MLOps project demonstrating:

**PyTorch → MLflow → Model Registry → FastAPI → Docker → GitHub Actions → Azure**

## Architecture

```text
PyTorch Training
      │
      ▼
MLflow Tracking
      │
      ▼
MLflow Model Registry
      │
      │ champion alias
      ▼
FastAPI Inference API
      │
      ▼
Docker Container
      │
      ▼
Azure Container Apps
```

## What this demonstrates

- PyTorch model training
- reproducible training configuration
- MLflow experiment tracking
- MLflow model artifacts
- model registry and versioning
- model aliases (`champion`)
- FastAPI model serving
- Pydantic validation
- Docker
- Pytest
- Ruff
- GitHub Actions CI/CD
- Azure Container Apps deployment

## Project structure

```text
mlops-model-serving-platform/
├── .github/workflows/
│   ├── ci.yml
│   └── deploy-azure.yml
├── app/
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── ml/
│   ├── data.py
│   ├── model.py
│   └── train.py
├── tests/
│   └── test_api.py
├── infra/
│   └── azure-container-apps.bicep
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## Local setup — Windows

Create the virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy environment configuration:

```powershell
Copy-Item .env.example .env
```

## Start MLflow

```powershell
docker compose up -d mlflow
```

Open:

```text
http://localhost:5000
```

## Train and register a model

```powershell
python -m ml.train
```

The training pipeline creates a small PyTorch classifier, logs parameters and validation metrics to MLflow, stores the model artifact, and registers a model version.

## Start FastAPI

```powershell
uvicorn app.main:app --reload
```

Open Swagger:

```text
http://localhost:8000/docs
```

Health:

```text
GET /health
```

Model information:

```text
GET /model
```

Prediction:

```text
POST /predict
```

Example:

```json
{
  "features": [0.2, -0.4, 0.7, 0.1]
}
```

## Docker

```powershell
docker compose up --build api
```

The API runs on:

```text
http://localhost:8000
```

MLflow runs on:

```text
http://localhost:5000
```

## Tests and quality

```powershell
pytest
ruff check .
ruff format .
```

## CI/CD

`ci.yml` runs linting, tests, and a Docker build.

`deploy-azure.yml` provides the deployment path:

```text
GitHub Actions
      ↓
Azure Container Registry
      ↓
Azure Container Apps
```

Production secrets belong in GitHub Actions Secrets, not in the repository.

## Azure

The Bicep template is in:

```text
infra/azure-container-apps.bicep
```

For a real production deployment, MLflow should use durable cloud storage and a managed metadata database rather than a local SQLite/volume setup.

## MLOps lifecycle

```text
Train
  ↓
Track experiment
  ↓
Evaluate model
  ↓
Register version
  ↓
Promote alias
  ↓
Serve with FastAPI
  ↓
Build container
  ↓
CI/CD
  ↓
Deploy to Azure
  ↓
Monitor
  ↓
Retrain
```

The API loads:

```text
models:/ticket_classifier@champion
```

rather than hard-coding a model version. This makes model promotion and rollback possible without changing serving code.

## Suggested Phase 2 upgrades

After the baseline works, add:

1. automated model quality gates
2. structured logging
3. prediction latency metrics
4. model version in every response/log
5. data validation
6. staging vs production model aliases
7. Azure Container Registry
8. Azure Container Apps
9. managed MLflow storage
10. automated retraining

## Portfolio value

This project is intended to demonstrate that you can build the **full ML lifecycle**, not just train a model:

**model development + experiment tracking + registry + serving + containers + CI/CD + cloud deployment.**


## Python 3.14 compatibility

This project is configured for Python 3.14.3 on the developer machine.
PyTorch is pinned to 2.9.1 for Python 3.14 compatibility.
The MLflow Docker service pins SQLAlchemy below 2.1 because MLflow 2.17.2
uses a pool class removed in SQLAlchemy 2.1.

### Windows / VS Code setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Start MLflow:

```powershell
docker compose down
docker compose up -d mlflow
docker compose ps
docker compose logs mlflow --tail 30
```

Open http://localhost:5000 after the MLflow container reports `Up`.

Train the model:

```powershell
python -m ml.train
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs
