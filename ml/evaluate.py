import numpy as np


def calculate_metrics(
    y_true: np.ndarray,
    y_probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate binary classification evaluation metrics."""

    y_true = np.asarray(y_true).astype(int)
    y_probabilities = np.asarray(y_probabilities)

    y_predicted = (y_probabilities >= 0.5).astype(int)

    true_positive = np.sum(
        (y_true == 1) & (y_predicted == 1)
    )
    true_negative = np.sum(
        (y_true == 0) & (y_predicted == 0)
    )
    false_positive = np.sum(
        (y_true == 0) & (y_predicted == 1)
    )
    false_negative = np.sum(
        (y_true == 1) & (y_predicted == 0)
    )

    accuracy = (
        true_positive + true_negative
    ) / len(y_true)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }