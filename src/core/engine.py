import os
from typing import Dict, Final, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import config, get_logger

logger = get_logger(__name__)

class InferenceEngine:
    """
    Production-grade ML Architecture for text classification.
    
    Handles hardware-accelerated inference with optimized resource management.
    """

    LABEL_MAP: Final[Dict[int, str]] = {
        0: "HUMAN-WRITTEN",
        1: "AI-GENERATED"
    }

    def __init__(self) -> None:
        """
        Initializes the engine with lazy-loading and hardware optimization.
        
        Pillar 1: Separation of concerns - Engine is decoupled from delivery layers.
        """
        self._device = self._select_device()
        self._tokenizer, self._model = self._load_assets()
        logger.info(f"Inference Engine online. Hardware: {self._device.type.upper()}")

    def _select_device(self) -> torch.device:
        """Adaptive hardware selection for CPU/GPU acceleration."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_assets(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """Secure loading and mounting of model weights."""
        if not os.path.exists(config.ADAPTER_PATH):
            logger.critical(f"Asset Integrity Failure: {config.ADAPTER_PATH} missing.")
            raise FileNotFoundError("Critical ML assets missing. Execution halted.")

        try:
            # Pinning revision to main for basic security, though specific commit hash is preferred for production
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, revision="main")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.MODEL_NAME, 
                num_labels=len(self.LABEL_MAP),
                revision="main"
            )
            
            # Atomic weight loading
            model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)
            model.to(self._device)
            model.eval()
            return tokenizer, model
        except Exception as e:
            logger.error("Failed to mount model weights into secure memory.")
            raise RuntimeError("Internal Inference Failure") from e

    def _sanitize(self, text: str) -> str:
        """Aggressive input sanitization and payload neutralization."""
        if not isinstance(text, str):
            raise ValueError("Malicious Payload Detected: Invalid data type.")

        # Byte-size truncation and control character eradication
        sanitized = text.strip()[:config.MAX_INPUT_CHARS]
        return "".join(char for char in sanitized if ord(char) >= 32 or char in "\n\r\t")

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float]:
        """
        High-performance classification with zero-trust guardrails.

        Args:
            text (str): Raw input text for analysis.

        Returns:
            Tuple[str, float]: Standardized label and softmax confidence.
        """
        clean_text = self._sanitize(text)
        if not clean_text:
            return "NULL_PAYLOAD", 0.0

        # Optimization - Minimal memory overhead tensors
        inputs = self._tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_INPUT_LENGTH
        ).to(self._device)

        outputs = self._model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        
        idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][idx].item()
        
        return self.LABEL_MAP[int(idx)], float(confidence)
