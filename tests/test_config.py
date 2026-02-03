# tests/test_config.py
import pytest
from pathlib import Path


def test_load_config_from_yaml(tmp_path):
    """Config loads values from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
server:
  host: "127.0.0.1"
  port: 8000

model:
  path: "test/model"
  dtype: "float16"
  cache_dir: "./cache"
  attn_implementation: "eager"

transcription:
  max_file_size_mb: 100
  timeout_seconds: 600
  default_max_new_tokens: 4096
""")

    from app.config import load_config

    config = load_config(config_file)

    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8000
    assert config.model.path == "test/model"
    assert config.model.dtype == "float16"
    assert config.transcription.max_file_size_mb == 100


def test_config_auto_dtype_detection():
    """Auto dtype returns appropriate type based on GPU."""
    from app.config import get_torch_dtype
    import torch

    # Test explicit values
    assert get_torch_dtype("float32") == torch.float32
    assert get_torch_dtype("float16") == torch.float16
    assert get_torch_dtype("bfloat16") == torch.bfloat16

    # Auto should return a valid dtype
    auto_dtype = get_torch_dtype("auto")
    assert auto_dtype in [torch.float16, torch.bfloat16, torch.float32]
