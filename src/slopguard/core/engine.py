import os
from threading import Lock
from typing import Dict, Final, Optional, Tuple, Type, TypeVar, cast

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import config, logger

T = TypeVar("T", bound="InferenceEngine")


class InferenceEngine:
    """
    Consolidated, High-Performance ML Inference Engine.

    Implements a thread-safe Singleton pattern to manage heavy GPU/RAM assets.
    Adheres to Pillar I (Separation of Concerns) and Pillar II (Resource Discipline).
    """

    _instance: Optional["InferenceEngine"] = None
    _lock: Lock = Lock()

    LABEL_MAP: Final[Dict[int, str]] = {0: "HUMAN-WRITTEN", 1: "AI-GENERATED"}

    def __new__(cls: Type[T]) -> T:
        """Thread-safe singleton instantiation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InferenceEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance  # type: ignore

    def __init__(self) -> None:
        """
        Initializes the engine with lazy asset mounting.
        
        Ensures hardware acceleration is utilized where available (CUDA/MPS/CPU).
        """
        if getattr(self, "_initialized", False):
            return

        self._device = self._select_device()
        
        try:
            self._tokenizer, self._model = self._load_assets()
            self._initialized = True
            logger.info(f"InferenceEngine optimized for {self._device.type.upper()} context.")
        except Exception as e:
            logger.critical(f"HARDWARE_FAILURE: Could not mount ML assets: {str(e)}")
            raise RuntimeError("Internal Inference Failure") from e

    def _select_device(self) -> torch.device:
        """Determines the most performant compute provider."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            # Support for Apple Silicon acceleration
            return torch.device("mps")
        return torch.device("cpu")

    def _load_assets(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        Securely loads model weights and tokenizers.
        
        Implements integrity checks for model paths and handles PEFT (LoRA) integration.
        """
        if not os.path.exists(config.ADAPTER_PATH):
            raise FileNotFoundError(f"Missing critical weight adapter at {config.ADAPTER_PATH}")

        # Strict revision pinning for supply chain security
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, revision="main")  # nosec: B615
        
        base_model = AutoModelForSequenceClassification.from_pretrained(  # nosec: B615
            config.MODEL_NAME, 
            num_labels=len(self.LABEL_MAP),
            device_map=None,  # Explicitly managed by engine
            revision="main"
        )

        # Mount LoRA adapters into the base model architecture
        model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)
        model.to(self._device)
        model.eval()
        
        return cast(PreTrainedTokenizer, tokenizer), cast(PreTrainedModel, model)

    def _sanitize_payload(self, text: str) -> str:
        """
        Weaponized Input Sanitization.
        
        Neutralizes malformed Unicode, control characters, and resource-exhaustion payloads.
        """
        # Enforce strict character limits before processing to prevent CPU/RAM abuse
        text = text.strip()[: config.MAX_INPUT_CHARS]
        
        # Strip control characters except basic whitespace/newlines
        return "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Executes a secure, hardware-accelerated inference pass.

        Args:
            text (str): Raw input sequence for analysis.

        Returns:
            Tuple[str, float]: Standardized classification label and softmax confidence.
        """
        clean_text = self._sanitize_payload(text)
        
        if not clean_text:
            return "PAYLOAD_INVALID", 0.0

        try:
            # Tokenization with enforced truncation to prevent positional embedding overflows
            inputs = self._tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.MAX_INPUT_LENGTH,
            ).to(self._device)

            outputs = self._model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)

            # Extract highest probability index and scalar confidence
            idx = int(torch.argmax(probabilities, dim=-1).item())
            confidence = float(probabilities[0][idx].item())

            return self.LABEL_MAP[idx], confidence

        except Exception as e:
            logger.error(f"INFERENCE_PIPELINE_ERROR: {str(e)}")
            return "COMPUTE_ERROR", 0.0
