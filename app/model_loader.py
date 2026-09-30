import os
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)
MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "ticket_classifier",
)
MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)


class ModelLoader:
    def __init__(self) -> None:
        self.model: Any | None = None

    def load(self) -> Any:
        if self.model is None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            self.model = mlflow.pyfunc.load_model(model_uri)
        return self.model

    def predict(self, features: list[float]) -> float:
        model = self.load()
        input_data = np.asarray([features], dtype=np.float32)
        prediction = model.predict(input_data)
        return float(np.asarray(prediction).reshape(-1)[0])


model_loader = ModelLoader()