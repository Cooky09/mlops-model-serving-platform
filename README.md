MLOps Model Serving Platform

Production-style machine learning model serving platform demonstrating an end-to-end MLOps workflow using PyTorch, MLflow, FastAPI, Docker, GitHub Actions, and Azure.

The project demonstrates the lifecycle of a machine learning model from deterministic training and experiment tracking through model registration, versioning, promotion, rollback, and API-based inference.

⸻

Overview

This project demonstrates how a trained machine learning model can move from development into a reproducible, containerized serving environment.

The platform currently implements:

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
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Promotion /   │
                    │     Rollback        │
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

⸻

What This Project Demonstrates

The project focuses on practical MLOps engineering patterns:

* PyTorch model development and training
* Deterministic dataset generation
* Reproducible training configuration
* MLflow experiment tracking
* Hyperparameter and metric logging
* MLflow model artifact management
* Model signatures
* Model registration and versioning
* MLflow model aliases
* Model promotion
* Model rollback
* Model lifecycle testing
* Dynamic model loading through a registry
* FastAPI model serving
* Request/response validation with Pydantic
* Docker containerization
* Docker Compose orchestration
* Automated testing with Pytest
* Code quality checks with Ruff
* CI/CD with GitHub Actions
* Infrastructure-as-code using Azure Bicep
* Production-oriented configuration using environment variables

⸻

Technology Stack

Area	Technology
Programming Language	Python
Machine Learning	PyTorch
Experiment Tracking	MLflow
Model Registry	MLflow Model Registry
API	FastAPI
Validation	Pydantic
Containerization	Docker
Local Orchestration	Docker Compose
Testing	Pytest
Code Quality	Ruff
CI/CD	GitHub Actions
Cloud	Microsoft Azure
Infrastructure	Azure Bicep
Version Control	Git / GitHub

⸻

Project Structure

mlops-model-serving-platform/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── deploy-azure.yml
│   └── pull_request_template.md
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
│   ├── evaluate.py
│   ├── lifecycle.py
│   ├── model.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_lifecycle.py
│   ├── test_model.py
│   └── test_model_persistence.py
│
├── infra/
│   └── azure-container-apps.bicep
│
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

⸻

Machine Learning Workflow

1. Dataset Generation

The project generates a deterministic binary classification dataset using NumPy.

The dataset contains four numerical features:

Feature 1
Feature 2
Feature 3
Feature 4

A fixed random seed is used so that dataset generation is reproducible across training runs.

The current configuration uses:

Random seed: 42
Samples:     1000
Features:    4

The generated data is split into:

80% Training
20% Validation

The dataset generation logic is implemented in:

ml/data.py

⸻

2. PyTorch Model

The classifier is implemented using PyTorch.

Architecture:

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

The current training configuration uses:

Epochs:          30
Learning rate:   0.01
Optimizer:       Adam

The model implementation is located in:

ml/model.py

⸻

Model Evaluation

Model evaluation is separated from the training implementation.

The evaluation module calculates classification metrics including:

Accuracy
Precision
Recall
F1 Score

Evaluation logic is implemented in:

ml/evaluate.py

Automated tests verify the metric calculations independently from the training pipeline.

⸻

MLflow Experiment Tracking

MLflow is used to track training experiments and manage model artifacts.

The training pipeline records configuration and performance information for each run.

Parameters

Examples include:

epochs
learning_rate
random_seed
n_samples

Metrics

The evaluation pipeline records classification metrics including:

accuracy
precision
recall
f1

Model Artifact

The trained PyTorch model is logged to MLflow with:

* Model signature
* Example input
* Serialized model artifact
* Model metadata
* Registered model name

The training implementation is located in:

ml/train.py

⸻

Model Registry

The trained model is registered in the MLflow Model Registry.

Example model:

ticket_classifier

Model versions allow multiple trained models to coexist.

For example:

ticket_classifier
│
├── Version 1
├── Version 2
└── Version 3

Each registered version represents a distinct model artifact produced by the training pipeline.

⸻

Model Lifecycle

The platform implements a model lifecycle that goes beyond simply storing model artifacts.

The lifecycle is:

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

The lifecycle supports:

Registration

A trained model is registered in MLflow and assigned a model version.

Promotion

The newly registered model version can be assigned to the configured production alias.

The default alias is:

production

Serving

The API can resolve a model through an alias instead of depending on a hard-coded model version.

For example:

models:/ticket_classifier@production

This allows the model version behind the alias to change without modifying application code.

Rollback

A production alias can be moved back to a previous registered model version.

The rollback helper is implemented in:

ml/lifecycle.py

This provides a simple mechanism for restoring a previous model version without rebuilding the serving application.

⸻

Model Lifecycle Testing

The model lifecycle is covered by automated tests.

The tests verify:

* Production alias movement
* Model promotion behaviour
* Model rollback behaviour
* Interaction with the MLflow client
* Lifecycle behaviour without requiring a live MLflow deployment

Lifecycle tests are located in:

tests/test_lifecycle.py

External MLflow interactions are mocked where appropriate so that the tests remain deterministic and fast.

⸻

Model Serving

The model is served through FastAPI.

The serving flow is:

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

The application separates model-serving logic from the training pipeline.

⸻

API Endpoints

Health Check

GET /health

Example response:

{
  "status": "healthy"
}

⸻

Model Information

GET /model

Example response:

{
  "model_name": "ticket_classifier",
  "model_alias": "production",
  "tracking_uri": "http://mlflow:5000"
}

⸻

Prediction

POST /predict

Example request:

{
  "features": [
    0.1,
    0.2,
    0.3,
    0.4
  ]
}

Example response:

{
  "prediction": 1,
  "probability": 0.5296782851219177,
  "model_name": "ticket_classifier",
  "model_alias": "production"
}

The API validates that exactly four numerical features are supplied.

⸻

Interactive API Documentation

Once the API is running, FastAPI provides interactive documentation.

Open:

http://localhost:8000/docs

The documentation can be used to test:

* Health checks
* Model metadata
* Prediction requests

⸻

Local Development

Prerequisites

Install:

* Python 3.14+
* Docker Desktop
* Git

Verify Python:

python --version

Verify Docker:

docker --version
docker compose version

⸻

Create Virtual Environment

Windows PowerShell:

python -m venv .venv

Activate:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

⸻

Running MLflow and FastAPI

The recommended local setup uses Docker Compose.

Start the services:

docker compose up --build

This starts:

FastAPI
    │
    └── http://localhost:8000
MLflow
    │
    └── http://localhost:5000

FastAPI documentation:

http://localhost:8000/docs

MLflow UI:

http://localhost:5000

⸻

Training the Model

The training script connects to the MLflow tracking server and creates a training run.

Run:

python -m ml.train

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
11. Assigns the configured alias to the registered version

⸻

Model Promotion

After training, MLflow creates a new model version.

The training pipeline assigns the configured alias to the registered version.

By default:

production

The serving application can resolve:

models:/ticket_classifier@production

rather than depending on a hard-coded version.

This separates:

Model training

from:

Model serving

and allows the production model version to change independently from the application code.

⸻

Model Rollback

If a newly promoted model needs to be replaced, the production alias can be moved to a previous version.

Conceptually:

Before rollback:
production → Version 3

After rollback:

production → Version 2

The API continues resolving:

models:/ticket_classifier@production

so no application code change is required.

This provides a simple model recovery mechanism at the registry level.

⸻

Docker

The API is packaged into a Docker image.

Build the API image:

docker compose build api

Start the application:

docker compose up

Check running containers:

docker compose ps

View API logs:

docker compose logs api

View MLflow logs:

docker compose logs mlflow

⸻

Testing

The project includes automated tests covering:

* API behaviour
* Dataset generation
* Evaluation metrics
* Model structure
* Model persistence
* Model lifecycle behaviour

Run the full test suite:

pytest

The test suite is designed to validate application behaviour without requiring every test to depend on a running external MLflow service.

⸻

Code Quality

Ruff is used for linting and code quality checks.

Run:

python -m ruff check .

The project uses pyproject.toml to centralize Python tooling configuration.

⸻

Engineering Workflow

Development follows a GitHub-based workflow:

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

The repository uses:

* GitHub Issues for work tracking
* Feature branches for changes
* Pull requests for review and integration
* GitHub Actions for automated validation
* Protected main branch
* Required CI checks before merging
* Issue-linked pull requests

The project documentation for contributors is available in:

CONTRIBUTING.md

⸻

CI/CD

GitHub Actions is used to automate validation and delivery workflows.

The CI workflow performs automated checks such as:

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

This provides an automated quality gate before changes are merged.

Deployment automation is defined separately from CI.

⸻

Azure Deployment

The project includes Azure infrastructure configuration using Bicep.

Infrastructure is defined in:

infra/azure-container-apps.bicep

The intended deployment architecture is:

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

This provides a path toward cloud-hosted model serving using infrastructure-as-code.

⸻

Configuration

Application configuration is controlled through environment variables.

Example:

MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_MODEL_NAME=ticket_classifier
MLFLOW_MODEL_ALIAS=production

The project includes:

.env.example

Sensitive or machine-specific configuration should not be committed to Git.

⸻

Engineering Decisions

Model Registry Instead of Bundling Models

The application does not package a specific trained model directly inside the API image.

Instead, the API retrieves the configured model from MLflow.

This keeps:

Application code

separate from:

Model artifacts

and allows models to be updated independently of the application.

⸻

Alias-Based Model Loading

The API uses an MLflow alias such as:

models:/ticket_classifier@production

instead of directly referencing:

models:/ticket_classifier/3

This allows the model version behind the production alias to change without requiring an application code change.

⸻

Model Rollback Through the Registry

Rollback is implemented by moving the production alias to a previous registered version.

This means rollback can occur at the model lifecycle layer without rebuilding the API image.

⸻

Deterministic Training

The training pipeline uses explicit dataset configuration and a fixed random seed.

This makes training behaviour more reproducible and provides a consistent foundation for automated testing.

⸻

Containerized Services

MLflow and FastAPI are separated into services for local development.

This creates a clearer boundary between:

MLOps platform services

and:

Inference services

⸻

Reproducibility

The project uses several practices to improve reproducibility:

* Deterministic dataset generation
* Explicit random seed
* Explicit dependency management
* Docker containerization
* Environment-based configuration
* MLflow experiment tracking
* Model versioning
* Model signatures
* Infrastructure-as-code
* Automated CI checks
* Automated test coverage

⸻

Current Engineering Status

The project is being developed incrementally through GitHub Issues and pull requests.

Phase 1 — Engineering Foundation

* GitHub development workflow
* Pull request templates
* Issue templates
* Contributor workflow
* Branch protection
* CI quality gates

Phase 2 — ML Pipeline

* Deterministic dataset generation
* Reproducible training configuration
* Model evaluation metrics
* Model artifact validation
* Model persistence validation

Phase 3 — Model Lifecycle

* MLflow model registration
* Model version management
* Model promotion
* Production alias management
* Model rollback
* Lifecycle integration testing

Planned Phases

Future work will extend the platform into:

Observability
     ↓
Data / Model Drift
     ↓
Deployment Automation
     ↓
Production Hardening

⸻

Future Improvements

Potential future improvements include:

Observability

* Prometheus metrics
* Grafana dashboards
* Structured logging
* Request latency monitoring
* Prediction monitoring
* Model health monitoring

Data and Model Monitoring

* Data drift detection
* Feature distribution monitoring
* Prediction drift detection
* Model performance monitoring
* Automated evaluation gates

Deployment

* Automated model promotion
* Canary deployments
* Blue/green deployments
* Production deployment workflows
* Azure Container Apps deployment automation

ML Platform Extensions

* Real-world ticket classification dataset
* Data versioning
* Feature store integration
* Distributed training
* GPU-based training
* Azure ML integration
* Kubernetes deployment

Security and Reliability

* API authentication
* Secrets management
* Network security
* Rate limiting
* Dependency vulnerability scanning
* Model access controls

⸻

Project Goal

The long-term goal of this project is to demonstrate a complete production-oriented MLOps platform:

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

The emphasis is on reproducibility, automation, testability, model lifecycle management, and production-oriented engineering practices rather than simply training a machine learning model.