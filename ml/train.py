import os

import mlflow
import mlflow.pytorch
import torch
from mlflow.models import infer_signature
from torch import nn
from torch.optim import Adam

from ml.data import make_dataset
from ml.model import build_model

EXPERIMENT_NAME = "ticket-classifier-v2"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ticket_classifier")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

EPOCHS = 30
LEARNING_RATE = 0.01
RANDOM_SEED = 42
N_SAMPLES = 1000

def main() -> None:
    # Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load dataset
    X_train, X_val, y_train, y_val = make_dataset(n_samples = N_SAMPLES,seed = RANDOM_SEED,)

    # Build model
    model = build_model()

    # Training configuration
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.BCELoss()

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    ).reshape(-1, 1)

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
    ).reshape(-1, 1)

    # Start MLflow run
    with mlflow.start_run() as run:
        # -----------------------------
        # Training
        # -----------------------------
        for epoch in range(EPOCHS):
            model.train()

            optimizer.zero_grad()

            predictions = model(X_train_tensor)

            loss = criterion(
                predictions,
                y_train_tensor,
            )

            loss.backward()

            optimizer.step()

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()

        with torch.no_grad():
            val_predictions = model(X_val_tensor)

            val_loss = criterion(
                val_predictions,
                y_val_tensor,
            )

            predicted_classes = (
                val_predictions >= 0.5
            ).float()

            accuracy = (
                predicted_classes == y_val_tensor
            ).float().mean().item()

        # -----------------------------
        # Log parameters
        # -----------------------------
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "random_seed": RANDOM_SEED,
                "n_samples": N_SAMPLES,
                "architecture": "4-16-8-1",
                "optimizer": "Adam",
                "loss_function": "BCELoss",
            }
        )

        # -----------------------------
        # Log metrics
        # -----------------------------
        mlflow.log_metrics(
            {
                "val_loss": val_loss.item(),
                "val_accuracy": accuracy,
            }
        )

        # -----------------------------
        # Create MLflow model signature
        # -----------------------------
        example_input = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            example_output = model(example_input)

        signature = infer_signature(
            example_input.numpy(),
            example_output.numpy(),
        )

        # -----------------------------
        # Log and register model
        # -----------------------------
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input.numpy(),
            signature=signature,
            registered_model_name=MODEL_NAME,
            serialization_format="pickle",
        )

        # -----------------------------
        # Print results
        # -----------------------------
        print(f"Model URI:      {model_info.model_uri}")
        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Run ID:             {run.info.run_id}")
        print(f"Validation loss:    {val_loss.item():.4f}")
        print(f"Validation accuracy:{accuracy:.4f}")
        print(f"Registered model:   {MODEL_NAME}")
        print("=" * 60)

if __name__ == "__main__":
    main()

