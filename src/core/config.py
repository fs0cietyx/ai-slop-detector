import logging
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Standard configuration schema for the AI Slop Detector.
    
    Enforces strict isolation and validation of environment variables.
    """
    
    # ML Engine Settings
    MODEL_NAME: str = Field(default="bert-base-uncased", description="Base transformer model")
    ADAPTER_PATH: str = Field(default="models/ai-slop-detector-v1", description="Path to LoRA adapters")
    MAX_INPUT_LENGTH: int = Field(default=512, ge=64, le=2048)
    MAX_INPUT_CHARS: int = Field(default=5000, ge=100, le=10000)
    
    # External API Settings
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    
    # Environment
    ENV: str = Field(default="production", pattern="^(development|staging|production)$")
    LOG_LEVEL: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @validator("GEMINI_API_KEY")
    def validate_api_key(cls, v, values):
        if values.get("ENV") == "production" and not v:
            logging.warning("GEMINI_API_KEY is missing. Generation features will be disabled.")
        return v

# Singleton instance for global access
config = AppConfig()

def get_logger(name: str) -> logging.Logger:
    """Configures enterprise-standard structured logging."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [AUDIT-ID: %(process)d] - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(config.LOG_LEVEL)
    return logger
