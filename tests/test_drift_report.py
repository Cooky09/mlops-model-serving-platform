from ml.drift_report import build_drift_report


def test_report_shows_no_drift_when_checks_are_clear():
    report = build_drift_report(
        data_drift=False,
        prediction_drift=False,
    )

    assert report == {
        "data_drift": False,
        "prediction_drift": False,
        "drift_detected": False,
    }


def test_report_shows_drift_when_data_drift_is_detected():
    report = build_drift_report(
        data_drift=True,
        prediction_drift=False,
    )

    assert report["data_drift"] is True
    assert report["prediction_drift"] is False
    assert report["drift_detected"] is True


def test_report_shows_drift_when_prediction_drift_is_detected():
    report = build_drift_report(
        data_drift=False,
        prediction_drift=True,
    )

    assert report["data_drift"] is False
    assert report["prediction_drift"] is True
    assert report["drift_detected"] is True