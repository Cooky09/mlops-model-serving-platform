from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_logs_request(caplog):
    with caplog.at_level("INFO"):
        response = client.get("/health")

    assert response.status_code == 200
    assert "Health check requested" in caplog.text

def test_model_metadata():
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json()["model_name"] == "ticket_classifier"
    assert response.json()["model_alias"] == "champion"
def test_model_info_logs_request(caplog):
    with caplog.at_level("INFO"):
        response = client.get("/model")

    assert response.status_code == 200
    assert "Model information requested" in caplog.text
    assert "ticket_classifier" in caplog.text

def test_prediction_requires_four_features():
    response = client.post("/predict", json={"features": [0.1, 0.2]})
    assert response.status_code == 422

@patch("app.main.model_loader.predict", return_value=0.8)
def test_prediction_logs_request(mock_predict, caplog):
    with caplog.at_level("INFO"):
        response = client.post(
            "/predict",
            json={"features": [0.1, 0.2, 0.3, 0.4]},
        )

    assert response.status_code == 200
    assert "Prediction requested" in caplog.text
    assert "Prediction completed" in caplog.text
    assert "ticket_classifier" in caplog.text
    mock_predict.assert_called_once_with([0.1, 0.2, 0.3, 0.4])

def test_metrics_endpoint_exposes_prometheus_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "api_request_latency_seconds" in response.text


def test_health_request_is_recorded_in_metrics():
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert (
        'api_requests_total{endpoint="/health",method="GET",status="200"}'
        in response.text
    )
    assert "api_request_latency_seconds" in response.text