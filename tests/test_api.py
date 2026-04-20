from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


def _mock_model() -> MagicMock:
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.9, 0.1]])
    return m


def _mock_preprocessor() -> MagicMock:
    m = MagicMock()
    m.transform.side_effect = lambda x: x
    return m


@pytest.fixture()
def client():
    """TestClient with joblib.load patched so the lifespan doesn't need real .pkl files."""
    with patch("joblib.load", side_effect=[_mock_model(), _mock_preprocessor()]):
        with TestClient(app) as c:
            yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_missing_required_fields(client):
    response = client.post("/predict", json={"TransactionAmt": 100.0})
    assert response.status_code == 422


def test_predict_negative_amount(client):
    payload = {
        "TransactionAmt": -50.0,
        "ProductCD": "W",
        "card1": 13926,
        "TransactionDT": 86400,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_response_shape(client):
    payload = {
        "TransactionAmt": 50.0,
        "ProductCD": "W",
        "card1": 13926,
        "TransactionDT": 86400,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert data["decision"] in ("APPROVED", "FLAGGED")
    assert 0.0 <= data["fraud_probability"] <= 1.0
