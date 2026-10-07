import numpy as np

from ml.drift import calculate_feature_drift


def test_no_drift_when_feature_means_are_similar():
    baseline = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )
    current = np.array(
        [
            [0.1, 1.1],
            [1.1, 2.1],
            [1.9, 2.9],
        ]
    )
    assert calculate_feature_drift(baseline, current) is False
def test_drift_when_feature_mean_changes():
    baseline = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )
    current = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    assert calculate_feature_drift(baseline, current) is True

def test_custom_threshold_controls_drift_detection():
    baseline = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )

    current = np.array(
        [
            [0.15, 1.0],
            [1.15, 2.0],
            [2.15, 3.0],
        ]
    )

    assert calculate_feature_drift(
        baseline,
        current,
        threshold=0.1,
    ) is True

    assert calculate_feature_drift(
        baseline,
        current,
        threshold=0.2,
    ) is False

def test_drift_detection_requires_matching_feature_dimensions():
    baseline = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ]
    )

    current = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ]
    )

    try:
        calculate_feature_drift(baseline, current)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for mismatched feature dimensions"
    )