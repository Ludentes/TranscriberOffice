# app/config.py
"""Configuration management for Meeting Transcriber."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 7860


@dataclass
class ModelConfig:
    path: str = "microsoft/VibeVoice-ASR"
    dtype: str = "auto"
    cache_dir: str = "./models"
    attn_implementation: str = "sdpa"


@dataclass
class TranscriptionConfig:
    max_file_size_mb: int = 500
    timeout_seconds: int = 1800
    default_max_new_tokens: int = 8192


@dataclass
class AppConfig:
    server: ServerConfig
    model: ModelConfig
    transcription: TranscriptionConfig


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype, with auto-detection."""
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "auto":
        # Auto-detect based on GPU capability
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Ampere (8.0+) supports bfloat16 well
            if capability[0] >= 8:
                return torch.bfloat16
            else:
                return torch.float16
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        # Return defaults if no config file
        return AppConfig(
            server=ServerConfig(),
            model=ModelConfig(),
            transcription=TranscriptionConfig()
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return AppConfig(
        server=ServerConfig(**data.get("server", {})),
        model=ModelConfig(**data.get("model", {})),
        transcription=TranscriptionConfig(**data.get("transcription", {}))
    )


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
