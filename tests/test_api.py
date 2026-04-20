from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# Patch artifact loading before the app module is imported
with patch("src.api.main.ARTIFACTS_DIR"):
    from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_missing_required_fields():
    response = client.post("/predict", json={"TransactionAmt": 100.0})
    assert response.status_code == 422


def test_predict_negative_amount():
    payload = {
        "TransactionAmt": -50.0,
        "ProductCD": "W",
        "card1": 13926,
        "TransactionDT": 86400,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@patch.dict(
    "src.api.main.model_store",
    {
        "model": MagicMock(predict_proba=MagicMock(return_value=np.array([[0.9, 0.1]]))),
        "preprocessor": MagicMock(transform=MagicMock(side_effect=lambda x: x)),
    },
)
def test_predict_approved_response():
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
