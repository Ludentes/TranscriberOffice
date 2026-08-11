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
    use_quantized: str = "auto"  # "auto", "true", "false"
    device_map: str = "auto"  # "auto" (multi-GPU if available), "single", or explicit like "balanced_low_0"


@dataclass
class TranscriptionConfig:
    max_file_size_mb: int = 500
    timeout_seconds: int = 1800
    default_max_new_tokens: int = 8192
    # Audio chunking settings
    # Set to 0 for auto-calculation based on GPU memory
    chunk_threshold_minutes: int = 0
    chunk_size_minutes: int = 0
    chunk_overlap_seconds: int = 10
    # Silence-based splitting (split at natural pauses instead of fixed intervals)
    silence_split: bool = True
    silence_noise_db: int = -30           # dB threshold for silence detection
    silence_min_duration: float = 0.5     # Minimum silence length in seconds
    silence_search_window: int = 30       # Seconds around target boundary to search for silence


@dataclass
class StorageConfig:
    data_dir: str = "./data"


@dataclass
class SessionConfig:
    cookie_name: str = "transcriber_session"
    cookie_secure: bool = False
    cookie_max_age_days: int = 365


@dataclass
class AppConfig:
    server: ServerConfig
    model: ModelConfig
    transcription: TranscriptionConfig
    storage: StorageConfig
    session: SessionConfig


def get_model_path(config: ModelConfig) -> str:
    """Resolve the model path based on use_quantized setting."""
    use_quantized = config.use_quantized.lower()

    if use_quantized == "true":
        return "scerz/VibeVoice-ASR-4bit"
    elif use_quantized == "false":
        return config.path
    elif use_quantized == "auto":
        # Auto-detect: use quantized for pre-Ampere GPUs
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:  # Pre-Ampere
                return "scerz/VibeVoice-ASR-4bit"
        return config.path
    else:
        return config.path


def is_quantized_model(config: ModelConfig) -> bool:
    """Check if the resolved model path is a quantized model."""
    return "4bit" in get_model_path(config).lower()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype, with auto-detection."""
    dtype_str = dtype_str.lower()
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


def auto_chunk_settings(quantized: bool = False) -> tuple[int, int]:
    """Calculate chunk threshold and size based on GPU memory.

    Uses the SMALLEST single-GPU free memory as the constraint,
    since audio encoding happens entirely on one GPU.

    Args:
        quantized: Whether the model is quantized (4-bit, ~4.5GB vs ~18GB)

    Returns:
        (chunk_threshold_minutes, chunk_size_minutes)
    """
    if not torch.cuda.is_available():
        return 5, 3  # Conservative defaults for CPU

    gpu_count = torch.cuda.device_count()

    # Get per-GPU memory
    gpu_memories_gb = [
        torch.cuda.get_device_properties(i).total_memory / (1024**3)
        for i in range(gpu_count)
    ]
    single_gpu_gb = min(gpu_memories_gb)  # Smallest GPU is the bottleneck

    # Model size depends on quantization
    model_gb = 5 if quantized else 18

    # Quantized model fits on 1 GPU; full model may be split across GPUs
    if quantized or gpu_count == 1:
        model_per_gpu_gb = model_gb
    else:
        model_per_gpu_gb = model_gb / gpu_count

    available_gb = single_gpu_gb - model_per_gpu_gb

    # Heuristic: ~2.0GB VRAM per minute of audio for inference activations
    gb_per_minute = 2.0
    max_chunk_minutes = max(2, int(available_gb / gb_per_minute))

    # Cap at reasonable values
    chunk_size = min(max_chunk_minutes, 30)  # Never more than 30 min
    chunk_threshold = chunk_size + 2  # Trigger chunking 2 min before max

    print(f"Auto chunk settings: {gpu_count} GPU(s), {single_gpu_gb:.0f}GB each, "
          f"model={model_gb}GB {'(4-bit)' if quantized else '(fp16)'}, "
          f"~{model_per_gpu_gb:.0f}GB model/GPU, ~{available_gb:.0f}GB free/GPU -> "
          f"threshold={chunk_threshold}min, chunk_size={chunk_size}min")

    return chunk_threshold, chunk_size


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        # Return defaults if no config file
        return AppConfig(
            server=ServerConfig(),
            model=ModelConfig(),
            transcription=TranscriptionConfig(),
            storage=StorageConfig(),
            session=SessionConfig(),
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return AppConfig(
        server=ServerConfig(**data.get("server", {})),
        model=ModelConfig(**data.get("model", {})),
        transcription=TranscriptionConfig(**data.get("transcription", {})),
        storage=StorageConfig(**data.get("storage", {})),
        session=SessionConfig(**data.get("session", {})),
    )


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
