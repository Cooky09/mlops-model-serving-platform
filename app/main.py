import logging
import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest

from app.model_loader import MODEL_ALIAS, MODEL_NAME, model_loader
from app.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
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
    start_time = time.perf_counter()

    logger.info("Health check requested")

    if not model_loader.is_healthy():
        logger.error("Model dependency is unhealthy")

        REQUEST_COUNT.labels(
            endpoint="/health",
            method="GET",
            status="503",
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/health",
            method="GET",
        ).observe(time.perf_counter() - start_time)

        raise HTTPException(
            status_code=503,
            detail="Model dependency unavailable",
        )

    response = {
        "status": "healthy",
    }

    REQUEST_COUNT.labels(
        endpoint="/health",
        method="GET",
        status="200",
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint="/health",
        method="GET",
    ).observe(time.perf_counter() - start_time)

    return response


@app.get("/model")
def model_info() -> dict[str, str]:
    start_time = time.perf_counter()

    logger.info(
        "Model information requested: model=%s alias=%s",
        MODEL_NAME,
        MODEL_ALIAS,
    )

    response = {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "tracking_uri": (
            "http://mlflow:5000"
            if MODEL_NAME
            else ""
        ),
    }

    REQUEST_COUNT.labels(
        endpoint="/model",
        method="GET",
        status="200",
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint="/model",
        method="GET",
    ).observe(time.perf_counter() - start_time)

    return response

@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    start_time = time.perf_counter()

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

        REQUEST_COUNT.labels(
            endpoint="/predict",
            method="POST",
            status="200",
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/predict",
            method="POST",
        ).observe(time.perf_counter() - start_time)

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

        REQUEST_COUNT.labels(
            endpoint="/predict",
            method="POST",
            status="500",
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/predict",
            method="POST",
        ).observe(time.perf_counter() - start_time)

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc