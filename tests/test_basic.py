import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.core.config import config
from src.core.engine import InferenceEngine

def test_config_validation():
    """Verify that configuration maintains strict type integrity."""
    assert config.APP_NAME == "AI-Slop-Detector"
    assert config.MAX_INPUT_LENGTH >= 64
    assert config.MAX_INPUT_CHARS <= 10000

def test_engine_singleton():
    """Verify that the InferenceEngine correctly implements the Singleton pattern."""
    # Reset singleton for testing
    InferenceEngine._instance = None
    
    with patch.object(InferenceEngine, '_load_assets', return_value=(MagicMock(), MagicMock())):
        engine1 = InferenceEngine()
        engine2 = InferenceEngine()
        assert engine1 is engine2

def test_input_sanitization():
    """Verify that weaponized payloads are correctly neutralized."""
    with patch.object(InferenceEngine, '_load_assets', return_value=(MagicMock(), MagicMock())):
        engine = InferenceEngine()
        
        # Test control character stripping
        malicious_payload = "Normal text\x00\x1Fwith control chars"
        sanitized = engine._sanitize_payload(malicious_payload)
        assert "\x00" not in sanitized
        assert "\x1F" not in sanitized
        
        # Test extreme truncation
        long_payload = "A" * 20000
        truncated = engine._sanitize_payload(long_payload)
        assert len(truncated) <= config.MAX_INPUT_CHARS

def test_api_health():
    """Verify that the API gateway health check is functional."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    # TestClient handles the internal event loop synchronously
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
