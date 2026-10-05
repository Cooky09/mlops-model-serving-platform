import logging

from fastapi import FastAPI, HTTPException

from app.model_loader import MODEL_ALIAS, MODEL_NAME, model_loader
from app.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)
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
    logger.info("Health check requested")
    return {
        "status": "healthy",
    }


@app.get("/model")
def model_info() -> dict[str, str]:
    logger.info(
        "Model information requested: model=%s alias=%s",
        MODEL_NAME,
        MODEL_ALIAS,
    )
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
    logger.info(
        "Prediction requested: model=%s alias=%s",
        MODEL_NAME,
        MODEL_ALIAS,
    )
    try:
        probability = model_loader.predict(
            request.features
        )

        prediction = int(probability >= 0.5)
        logger.info(
            "Prediction completed: model=%s alias=%s prediction=%s",
            MODEL_NAME,
            MODEL_ALIAS,
            prediction,
        )

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )

    except Exception as exc:
        logger.exception(
            "Prediction failed: model=%s alias=%s",
            MODEL_NAME,
            MODEL_ALIAS,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc
