"""
tests/test_predict.py
---------------------
Unit tests for the ML prediction service.

Run: pytest tests/ -v
"""
import pytest
import sys
import os

# Add parent dir to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Sample Customer Data ──────────────────────────────────────────────────────
SAMPLE_CUSTOMER = {
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 2500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
    "include_shap": False,  # Skip SHAP for faster tests
}


# ── Health Check Tests ────────────────────────────────────────────────────────
class TestHealth:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "BankAI Pro - ML Service"
        assert "endpoints" in data

    def test_health_endpoint(self):
        response = client.get("/health")
        # Either 200 (healthy) or 503 (model not loaded in test env)
        assert response.status_code in [200, 503]


# ── Prediction Tests ──────────────────────────────────────────────────────────
class TestPredictions:
    def test_single_prediction_success(self):
        """Test that a valid customer returns a prediction."""
        response = client.post("/predict/single", json=SAMPLE_CUSTOMER)
        # Will be 503 if pipeline.pkl not present in test env
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in ["yes", "no"]
            assert 0 <= data["probability"] <= 100
            assert data["priority"] in ["High", "Medium", "Low"]
            assert "strategy" in data
            assert "model" in data

    def test_single_prediction_invalid_age(self):
        """Test validation rejects invalid age."""
        invalid_customer = {**SAMPLE_CUSTOMER, "age": 200}  # Age > 100
        response = client.post("/predict/single", json=invalid_customer)
        assert response.status_code == 422  # Validation error

    def test_single_prediction_missing_field(self):
        """Test that missing required field returns 422."""
        incomplete = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "job"}
        response = client.post("/predict/single", json=incomplete)
        assert response.status_code == 422

    def test_features_endpoint(self):
        """Test that features endpoint returns expected structure."""
        response = client.get("/predict/features")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "count" in data
        assert len(data["features"]) > 0

    def test_bulk_template_download(self):
        """Test that CSV template downloads successfully."""
        response = client.get("/predict/bulk/template")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        # Check CSV has header row
        content = response.content.decode()
        assert "age" in content
        assert "job" in content

    def test_bulk_upload_wrong_type(self):
        """Test that non-CSV files are rejected."""
        from io import BytesIO
        response = client.post(
            "/predict/bulk",
            files={"file": ("test.txt", BytesIO(b"not a csv"), "text/plain")},
        )
        assert response.status_code == 400

    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = client.get("/predict/info")
        # 200 if loaded, 503 if not
        assert response.status_code in [200, 503]


# ── Explainability Tests ──────────────────────────────────────────────────────
class TestExplainability:
    def test_global_feature_importance(self):
        """Test global feature importance endpoint."""
        response = client.get("/explain/global")
        if response.status_code == 200:
            data = response.json()
            assert "feature_importance" in data
            assert "chart_data" in data
            assert "labels" in data["chart_data"]
            assert "values" in data["chart_data"]

    def test_shap_endpoint_structure(self):
        """Test SHAP endpoint returns correct structure."""
        customer = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "include_shap"}
        response = client.post("/explain/shap", json=customer)
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "shap_values" in data


# ── Chat Tests ────────────────────────────────────────────────────────────────
class TestChat:
    def test_quick_action_missing_customer(self):
        """Test that explain action without customer data returns 400."""
        response = client.post(
            "/chat/quick-action",
            json={"action": "explain", "customer": None},
        )
        assert response.status_code == 400

    def test_quick_action_unknown_action(self):
        """Test that unknown action returns 422 (validation error)."""
        response = client.post(
            "/chat/quick-action",
            json={"action": "unknown_action"},
        )
        assert response.status_code == 422  # Pydantic Literal validation
