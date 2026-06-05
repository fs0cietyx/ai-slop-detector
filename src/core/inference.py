import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
from peft import PeftModel
from typing import Tuple, Dict, Optional, Union
import os
import logging
from dataclasses import dataclass

# Pillar I: Architectural Rigor - Dedicated Config Container
@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the AI Slop Detector."""
    model_dir: str
    base_model: str = "bert-base-uncased"
    max_length: int = 512
    max_input_chars: int = 5000

class DeviceProvider:
    """Pillar III: Non-Blocking Resource Management - Encapsulates hardware selection."""
    
    @staticmethod
    def get_optimal_device() -> torch.device:
        """Determines the most performant available hardware device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

class AISlopDetector:
    """
    Principal ML Architecture for AI-generated text classification.
    
    Adheres to Pillar I (Architectural Rigor) and Pillar IV (Defensive Programming).
    """

    def __init__(self, config: DetectorConfig):
        """
        Initializes the detector with a specific configuration.

        Args:
            config (DetectorConfig): The immutable configuration for the detector.

        Raises:
            FileNotFoundError: If the model directory is invalid.
            RuntimeError: If model weights fail to load onto the target device.
        """
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._device = DeviceProvider.get_optimal_device()
        
        if not os.path.isdir(self._config.model_dir):
            raise FileNotFoundError(f"Model directory not found at {self._config.model_dir}")

        try:
            self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self._config.base_model)
            
            # Pillar II: ML Optimization - Lazy loading of base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self._config.base_model, 
                num_labels=2
            )
            
            # Load and move to device in one atomic operation
            self._model = PeftModel.from_pretrained(base_model, self._config.model_dir)
            self._model.to(self._device)
            self._model.eval()
            
            self._logger.info(f"Detector initialized on {self._device}")
        except Exception as e:
            self._logger.critical(f"Critical failure during model initialization: {str(e)}")
            raise RuntimeError("Model loading failed.") from e

    def _sanitize_input(self, text: str) -> str:
        """
        Pillar IV: Defensive Programming - Aggressive input sanitization.
        
        Args:
            text (str): Raw input text.

        Returns:
            str: Sanitized and truncated text.
        """
        if not isinstance(text, str):
            raise ValueError("Input payload must be a string sequence.")

        # Truncate early to prevent resource amplification
        text = text.strip()[:self._config.max_input_chars]
        
        # Remove null bytes and non-printable control characters
        return "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Executes a secure inference pass.

        Args:
            text (str): The text content to analyze.

        Returns:
            Tuple[str, float]: A tuple containing the label (AI-GENERATED/HUMAN-WRITTEN) 
                and the confidence score (0.0 to 1.0).
        """
        sanitized_text = self._sanitize_input(text)
        if not sanitized_text:
            return "NULL_PAYLOAD", 0.0

        try:
            # Pillar II: NLP Optimization - Efficient tensor allocation
            inputs: Dict[str, torch.Tensor] = self._tokenizer(
                sanitized_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self._config.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                
            prediction_idx = torch.argmax(probabilities, dim=-1).item()
            confidence: float = probabilities[0][prediction_idx].item()
            
            label = "AI-GENERATED" if prediction_idx == 1 else "HUMAN-WRITTEN"
            return label, confidence

        except Exception as e:
            self._logger.error(f"Inference pipeline failure: {str(e)}")
            return "PIPELINE_ERROR", 0.0
