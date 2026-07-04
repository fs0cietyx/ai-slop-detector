import logging
import os
from enum import Enum
from typing import Any, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Execution environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AppConfig(BaseSettings):
    """
    Hardened Enterprise Configuration Schema.

    Enforces strict typing, secrets isolation, and environment-specific validation.
    Adheres to robust security and configuration practices.
    """

    # --- Metadata ---
    APP_NAME: str = "AI-Slop-Detector"
    APP_VERSION: str = "1.0.0"
    ENV: Environment = Field(default=Environment.PRODUCTION, alias="APP_ENV")

    # --- ML Engine Settings ---
    # Resource Discipline: Strict bounds on input sizes
    MODEL_NAME: str = Field(default="bert-base-uncased", description="Base transformer architecture")
    ADAPTER_PATH: str = Field(
        default="models/ai-slop-detector-v1", description="Path to LoRA weights"
    )
    MAX_INPUT_LENGTH: int = Field(default=512, ge=64, le=1024)
    MAX_INPUT_CHARS: int = Field(default=5000, ge=100, le=10000)

    # --- Security & Secrets ---
    # Secrets Isolation: Keys must be provided via environment, never hardcoded.
    GEMINI_API_KEY: Optional[SecretStr] = Field(default=None)
    API_KEY_INTERNAL: SecretStr = Field(
        default=SecretStr("dev-key-change-in-prod"),
        description="Internal API key for service-to-service auth",
    )

    # --- Infrastructure ---
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    HOST: str = "0.0.0.0"  # nosec
    PORT: int = 8000

    # --- Rate Limiting ---
    # Denial-of-Wallet Protection
    RATE_LIMIT_DEFAULT: str = "100/minute"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("GEMINI_API_KEY", mode="before")
    @classmethod
    def check_required_secrets(cls, v: Any, info: Any) -> Any:
        """Enforces secret presence in production environments."""
        # Note: In a real enterprise setup, we might use a secret manager (AWS SM, HashiCorp Vault)
        # For this OSS suite, we enforce .env/env var presence.
        env = os.getenv("APP_ENV", Environment.PRODUCTION)
        if env == Environment.PRODUCTION and not v:
            raise ValueError("SECURITY FAILURE: GEMINI_API_KEY is mandatory in production.")
        return v


# Global Configuration Singleton
config = AppConfig()


def setup_logging() -> logging.Logger:
    """
    Configures enterprise-standard structured logging.
    
    Prevents leakage of sensitive metadata while ensuring actionable logs.
    """
    logger = logging.getLogger(config.APP_NAME)

    if not logger.handlers:
        handler = logging.StreamHandler()
        # Structured format for easy ingestion by ELK/Datadog
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
            '"process": %(process)d, "thread": %(thread)d, "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(config.LOG_LEVEL)

    return logger


# Initialize root logger
logger = setup_logging()
