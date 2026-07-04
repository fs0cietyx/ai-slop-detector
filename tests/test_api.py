from unittest.mock import patch

from fastapi.testclient import TestClient
from slopguard.api.main import app
from slopguard.core.config import Environment, config
from slopguard.core.engine import InferenceEngine

client = TestClient(app)


def test_api_health() -> None:
    """Verify that the API gateway health check is functional."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "engine_online" in data


def test_predict_success() -> None:
    """Test successful API prediction using mocked engine."""
    with patch.object(InferenceEngine, "predict", return_value=("AI-GENERATED", 0.95)):
        response = client.post(
            "/v1/predict",
            json={
                "text": "This is a sufficiently long payload to pass the validation constraints."
            },
            headers={"X-API-KEY": config.API_KEY_INTERNAL.get_secret_value()},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "AI-GENERATED"
        assert data["confidence"] == 0.95
        assert data["status"] == "SUCCESS"
        assert "latency_ms" in data


def test_predict_invalid_payload() -> None:
    """Test API rejects payloads that are too short."""
    response = client.post(
        "/v1/predict",
        json={"text": "Too short"},
        headers={"X-API-KEY": config.API_KEY_INTERNAL.get_secret_value()},
    )
    assert response.status_code == 422  # Unprocessable Entity (Pydantic Validation)


def test_predict_unauthorized_in_production() -> None:
    """Test API enforces authentication in production mode."""
    # Temporarily set ENV to production
    original_env = config.ENV
    config.ENV = Environment.PRODUCTION

    try:
        response = client.post(
            "/v1/predict",
            json={
                "text": "This is a sufficiently long payload to pass the validation constraints."
            },
            # Missing X-API-KEY header
        )
        assert response.status_code == 401
    finally:
        config.ENV = original_env


def test_predict_internal_error() -> None:
    """Test API gracefully handles engine crashes."""
    with patch.object(InferenceEngine, "predict", side_effect=Exception("Critical engine failure")):
        response = client.post(
            "/v1/predict",
            json={
                "text": "This is a sufficiently long payload to pass the validation constraints."
            },
            headers={"X-API-KEY": config.API_KEY_INTERNAL.get_secret_value()},
        )
        assert response.status_code == 500
        assert "internal error" in response.json()["detail"].lower()
