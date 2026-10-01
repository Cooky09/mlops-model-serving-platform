import numpy as np

from ml.evaluate import calculate_metrics


def test_calculate_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_metrics(
        y_true,
        y_probabilities,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_calculate_metrics_with_mixed_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_probabilities = np.array([0.2, 0.7, 0.8, 0.3])

    metrics = calculate_metrics(
        y_true,
        y_probabilities,
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5