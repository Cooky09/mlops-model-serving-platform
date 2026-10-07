import numpy as np


def calculate_feature_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.1,
) -> bool:
    """Detect whether the current feature distribution has drifted."""
    if baseline.ndim != 2 or current.ndim != 2:
        raise ValueError("Baseline and current data must be 2-dimensional")
    if baseline.shape[1] != current.shape[1]:
        raise ValueError(
            "Baseline and current data must have the same number of features"
        )
    baseline_mean = np.mean(baseline, axis=0)
    current_mean = np.mean(current, axis=0)
    mean_difference = np.abs(current_mean - baseline_mean)
    return bool(np.any(mean_difference > threshold))