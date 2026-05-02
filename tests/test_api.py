# tests/test_api.py
from fastapi.testclient import TestClient
from src.api import app

def test_health_check():
    # 'with' ensures startup logic completes
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_prediction():
    with TestClient(app) as client:
        payload = {"description": "Chocolate Sandwich Cookies"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "predicted_category" in response.json()