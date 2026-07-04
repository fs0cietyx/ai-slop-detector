import pytest
from unittest.mock import MagicMock, patch
import torch

from slopguard.core.config import config
from slopguard.core.engine import InferenceEngine


def test_config_validation() -> None:
    """Verify that configuration maintains strict type integrity."""
    assert config.APP_NAME == "AI-Slop-Detector"
    assert config.MAX_INPUT_LENGTH >= 64
    assert config.MAX_INPUT_CHARS <= 10000


def test_engine_singleton() -> None:
    """Verify that the InferenceEngine correctly implements the Singleton pattern."""
    InferenceEngine._instance = None

    with patch.object(InferenceEngine, "_load_assets", return_value=(MagicMock(), MagicMock())):
        engine1 = InferenceEngine()
        engine2 = InferenceEngine()
        assert engine1 is engine2


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Normal text", "Normal text"),
        ("Dirty \x00 text", "Dirty  text"),
        ("New\nlines\tare ok", "New\nlines\tare ok"),
        ("\x1fControl chars removed", "Control chars removed"),
        ("A" * (config.MAX_INPUT_CHARS + 100), "A" * config.MAX_INPUT_CHARS),
    ],
)
def test_input_sanitization(input_text: str, expected: str) -> None:
    """Verify that payloads are correctly neutralized."""
    InferenceEngine._instance = None
    with patch.object(InferenceEngine, "_load_assets", return_value=(MagicMock(), MagicMock())):
        engine = InferenceEngine()
        sanitized = engine._sanitize_payload(input_text)
        assert sanitized == expected


def test_predict_empty_payload() -> None:
    """Ensure empty or fully invalid payloads are rejected safely."""
    InferenceEngine._instance = None
    with patch.object(InferenceEngine, "_load_assets", return_value=(MagicMock(), MagicMock())):
        engine = InferenceEngine()
        label, conf = engine.predict("   \x00   ")
        assert label == "PAYLOAD_INVALID"
        assert conf == 0.0


def test_predict_success() -> None:
    """Test successful hardware-accelerated prediction using a mocked model."""
    InferenceEngine._instance = None

    mock_tokenizer = MagicMock()
    # Mock tokenization to return a dummy tensor
    mock_tokenizer.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    mock_model = MagicMock()
    # Mock logits output: [batch_size, num_classes]
    # Logits: class 1 is higher than class 0, so it should predict AI-GENERATED (1)
    mock_model.return_value.logits = torch.tensor([[-1.0, 5.0]])

    with patch.object(InferenceEngine, "_load_assets", return_value=(mock_tokenizer, mock_model)):
        engine = InferenceEngine()
        label, conf = engine.predict("This is some AI generated slop text")
        
        assert label == "AI-GENERATED"
        # 5.0 vs -1.0 softmax means class 1 is ~99.7%
        assert conf > 0.9

def test_predict_error_handling() -> None:
    """Ensure the engine gracefully degrades when the model fails."""
    InferenceEngine._instance = None
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = Exception("Out of memory error")
    
    with patch.object(InferenceEngine, "_load_assets", return_value=(mock_tokenizer, MagicMock())):
        engine = InferenceEngine()
        label, conf = engine.predict("Valid text but boom")
        assert label == "COMPUTE_ERROR"
        assert conf == 0.0
