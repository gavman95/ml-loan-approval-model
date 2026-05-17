import numpy as np
from fastapi.testclient import TestClient

import src.fast_app.fast_app as fast_app_module


class _FakePredictionResult:
    predictions_labels = np.array(["Approved"])


def test_predict_endpoint_returns_predictions(monkeypatch):
    def _fake_pipeline(_df):
        return _FakePredictionResult()

    monkeypatch.setattr(fast_app_module, "model_predictions_pipeline", _fake_pipeline)
    client = TestClient(fast_app_module.app)

    payload = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "Yes",
        "income_annum": 100000,
        "loan_amount": 20000,
        "loan_term": 12,
        "cibil_score": 700,
        "residential_assets_value": 1000,
        "commercial_assets_value": 2000,
        "luxury_assets_value": 3000,
        "bank_asset_value": 4000,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predictions": ["Approved"]}
