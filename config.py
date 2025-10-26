"""
Configuration management for batch processing toolkit.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API configuration settings."""
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5-20250929"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 10
    polling_interval: int = 10
    max_retries: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    console_output: bool = True


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("batch_processing")
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.file_path:
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config() -> APIConfig:
    """Load configuration from environment variables."""
    return APIConfig(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20250929"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "10")),
        polling_interval=int(os.getenv("POLLING_INTERVAL", "10")),
        max_retries=int(os.getenv("MAX_RETRIES", "3"))
    )


def validate_config(config: APIConfig) -> None:
    """Validate configuration settings."""
    if not config.anthropic_api_key and not config.openai_api_key:
        raise ValueError("At least one API key must be provided")
    
    if config.temperature < 0 or config.temperature > 2:
        raise ValueError("Temperature must be between 0 and 2")
    
    if config.max_tokens < 1 or config.max_tokens > 1000:
        raise ValueError("Max tokens must be between 1 and 1000")
    
    if config.polling_interval < 1:
        raise ValueError("Polling interval must be at least 1 second")
