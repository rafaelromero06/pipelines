"""Tests de app/api.py -- corridos por pytest en .github/workflows/ci.yml (Etapa 5)."""
from fastapi.testclient import TestClient

from app.api import FEATURE_ORDER, app

client = TestClient(app)

SAMPLE_HIGH_RISK = [46, 115, 0, 0, 113, 1.5, "M", "ASY", "Normal", "Y", "Flat"]
SAMPLE_LOW_RISK = [40, 140, 289, 0, 172, 0.0, "M", "ATA", "Normal", "N", "Up"]


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_feature_order_has_11_columns():
    assert len(FEATURE_ORDER) == 11


def test_predict_returns_valid_schema():
    r = client.post("/predict", json={"features": SAMPLE_HIGH_RISK})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"prediction", "heart_disease_probability", "label"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["heart_disease_probability"] <= 1.0


def test_predict_high_risk_patient():
    r = client.post("/predict", json={"features": SAMPLE_HIGH_RISK})
    assert r.status_code == 200
    assert r.json()["prediction"] == 1
    assert r.json()["label"] == "Con enfermedad"


def test_predict_low_risk_patient():
    r = client.post("/predict", json={"features": SAMPLE_LOW_RISK})
    assert r.status_code == 200
    assert r.json()["prediction"] == 0
    assert r.json()["label"] == "Sin enfermedad"


def test_predict_wrong_number_of_features():
    r = client.post("/predict", json={"features": [1, 2, 3]})
    assert r.status_code == 422
