from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_metadata():
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json()["model_name"] == "ticket_classifier"
    assert response.json()["model_alias"] == "champion"


def test_prediction_requires_four_features():
    response = client.post("/predict", json={"features": [0.1, 0.2]})
    assert response.status_code == 422
