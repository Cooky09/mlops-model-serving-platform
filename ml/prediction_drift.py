import numpy as np


def calculate_prediction_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.1,
) -> bool:
    """Detect whether the distribution of predictions has drifted."""

    if baseline.size == 0 or current.size == 0:
        raise ValueError(
            "Baseline and current predictions must not be empty"
        )

    baseline_mean = np.mean(baseline)
    current_mean = np.mean(current)

    mean_difference = abs(current_mean - baseline_mean)

    return bool(mean_difference > threshold)