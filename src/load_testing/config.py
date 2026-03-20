"""Configuration loader for load testing."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

from models.scaledown_client import SCALEDOWN_SUPPORTED_MODELS


@dataclass
class ScaleDownConfig:
    """ScaleDown API configuration."""
    api_url: str
    api_key: str
    timeout: int = 30
    max_retries: int = 2
    retry_delay: float = 2.0


@dataclass
class DataConfig:
    """Data sampling configuration."""
    dataset: str
    split: str = "validation"
    num_samples: int = 100
    seed: int = 42


@dataclass
class LoadConfig:
    """Load test parameters."""
    scenario: str = "steady"
    default_users: int = 10
    default_spawn_rate: int = 2
    default_duration: int = 300  # seconds


@dataclass
class CompressionConfig:
    """Compression settings."""
    rates: List[Union[str, float]] = field(default_factory=lambda: ["auto"])
    models: List[str] = field(default_factory=lambda: ["gemini-2.5-flash"])


@dataclass
class OutputConfig:
    """Output settings."""
    raw_logs: bool = False
    results_dir: str = "load_testing/results"


@dataclass
class LoadTestConfig:
    """Complete load test configuration."""
    scaledown: ScaleDownConfig
    data: DataConfig
    load: LoadConfig
    compression: CompressionConfig
    output: OutputConfig


def _substitute_env_vars(value: str) -> str:
    """Substitute environment variables in string values.
    
    Replaces ${VAR_NAME} with environment variable value.
    Falls back to settings defaults if env var not set.
    
    Args:
        value: String potentially containing ${VAR_NAME} placeholders
        
    Returns:
        String with environment variables substituted
    """
    if not isinstance(value, str):
        return value
    
    # Pattern to match ${VAR_NAME}
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match):
        var_name = match.group(1)
        # Try environment variable first
        env_value = os.getenv(var_name)
        if env_value:
            return env_value
        
        # Fall back to settings defaults
        from settings import settings
        if hasattr(settings, var_name):
            return getattr(settings, var_name)
        
        # If not found, keep original placeholder
        return match.group(0)
    
    return re.sub(pattern, replacer, value)


def _validate_compression_rate(rate: Union[str, float]) -> None:
    """Validate compression rate value.
    
    Args:
        rate: Compression rate (must be "auto" or float 0.0-1.0)
        
    Raises:
        ValueError: If rate is invalid
    """
    if isinstance(rate, str):
        if rate != "auto":
            raise ValueError(
                f"String compression rate must be 'auto', got '{rate}'"
            )
    elif isinstance(rate, (int, float)):
        if not (0.0 <= rate <= 1.0):
            raise ValueError(
                f"Numeric compression rate must be between 0.0 and 1.0, got {rate}"
            )
    else:
        raise ValueError(
            f"Compression rate must be 'auto' or a number, got {type(rate)}"
        )


def _validate_compression_model(model: str) -> None:
    """Validate compression model against supported models.
    
    Args:
        model: Model name to validate
        
    Raises:
        ValueError: If model is not supported
    """
    # Handle "provider:model" format
    if ":" in model:
        parts = model.split(":")
        model_name = parts[1] if len(parts) > 1 else parts[0]
    else:
        model_name = model
    
    if model_name not in SCALEDOWN_SUPPORTED_MODELS:
        raise ValueError(
            f"Compression model '{model}' is not supported by ScaleDown. "
            f"Supported models: {sorted(SCALEDOWN_SUPPORTED_MODELS)}"
        )


def load_config(config_path: str = "load_testing/config.yaml") -> LoadTestConfig:
    """Load and validate load test configuration.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated LoadTestConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        raw_config = yaml.safe_load(f)
    
    # Substitute environment variables in scaledown section
    if "scaledown" in raw_config:
        for key in ["api_url", "api_key"]:
            if key in raw_config["scaledown"]:
                raw_config["scaledown"][key] = _substitute_env_vars(
                    raw_config["scaledown"][key]
                )
    
    # Validate compression rates
    if "compression" in raw_config and "rates" in raw_config["compression"]:
        for rate in raw_config["compression"]["rates"]:
            _validate_compression_rate(rate)
    
    # Validate compression models
    if "compression" in raw_config and "models" in raw_config["compression"]:
        for model in raw_config["compression"]["models"]:
            _validate_compression_model(model)
    
    # Build dataclass instances
    scaledown_config = ScaleDownConfig(**raw_config.get("scaledown", {}))
    data_config = DataConfig(**raw_config.get("data", {}))
    load_config = LoadConfig(**raw_config.get("load", {}))
    compression_config = CompressionConfig(**raw_config.get("compression", {}))
    output_config = OutputConfig(**raw_config.get("output", {}))
    
    # Validate required fields
    if not scaledown_config.api_key or scaledown_config.api_key.startswith("${"):
        raise ValueError(
            "SCALEDOWN_API_KEY not set. Please set environment variable or update config."
        )
    
    if not scaledown_config.api_url or scaledown_config.api_url.startswith("${"):
        raise ValueError(
            "SCALEDOWN_API_URL not set. Please set environment variable or update config."
        )
    
    return LoadTestConfig(
        scaledown=scaledown_config,
        data=data_config,
        load=load_config,
        compression=compression_config,
        output=output_config,
    )
