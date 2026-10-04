import mlflow
from mlflow import MlflowClient


def rollback_model(
    model_name: str,
    target_version: str,
    alias: str = "production",
) -> None:
    """Move a model alias to a specified registered model version."""

    client = MlflowClient(
        tracking_uri=mlflow.get_tracking_uri(),
    )

    client.set_registered_model_alias(
        model_name,
        alias,
        target_version,
    )