from typing import Any


def build_drift_report(
    data_drift: bool,
    prediction_drift: bool,
) -> dict[str, Any]:
    """Build a machine-readable report from drift detection results."""

    return {
        "data_drift": data_drift,
        "prediction_drift": prediction_drift,
        "drift_detected": data_drift or prediction_drift,
    }