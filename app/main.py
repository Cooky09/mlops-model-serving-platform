from fastapi import FastAPI, HTTPException

from app.model_loader import (
    MODEL_ALIAS,
    MODEL_NAME,
    model_loader,
)
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="MLOps Model Serving API",
    description=(
        "Production-style ML model serving API using "
        "FastAPI and MLflow."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@app.get("/model")
def model_info() -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "tracking_uri": (
            "http://mlflow:5000"
            if MODEL_NAME
            else ""
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    try:
        probability = model_loader.predict(
            request.features
        )

        prediction = int(probability >= 0.5)

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc