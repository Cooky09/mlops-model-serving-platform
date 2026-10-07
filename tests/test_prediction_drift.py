import numpy as np

from ml.prediction_drift import calculate_prediction_drift


def test_no_prediction_drift_when_means_are_similar():
    baseline = np.array([0.4, 0.5, 0.6])
    current = np.array([0.45, 0.5, 0.55])

    assert calculate_prediction_drift(baseline, current) is False


def test_prediction_drift_when_mean_changes():
    baseline = np.array([0.2, 0.3, 0.4])
    current = np.array([0.7, 0.8, 0.9])

    assert calculate_prediction_drift(baseline, current) is True


def test_custom_threshold_controls_prediction_drift():
    baseline = np.array([0.4, 0.5, 0.6])
    current = np.array([0.55, 0.65, 0.75])

    assert calculate_prediction_drift(
        baseline,
        current,
        threshold=0.1,
    ) is True

    assert calculate_prediction_drift(
        baseline,
        current,
        threshold=0.2,
    ) is False

def test_prediction_drift_rejects_empty_predictions():
    baseline = np.array([])
    current = np.array([0.4, 0.5, 0.6])

    try:
        calculate_prediction_drift(baseline, current)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for empty baseline predictions"
    )