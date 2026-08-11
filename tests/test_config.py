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


def test_load_config_empty_file(tmp_path):
    """Config handles empty YAML file gracefully."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")

    from app.config import load_config

    config = load_config(config_file)

    # Should return defaults when file is empty
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 7860
    assert config.model.path == "microsoft/VibeVoice-ASR"
    assert config.transcription.max_file_size_mb == 500


def test_get_torch_dtype_case_insensitive():
    """get_torch_dtype handles mixed case input."""
    from app.config import get_torch_dtype
    import torch

    assert get_torch_dtype("FLOAT32") == torch.float32
    assert get_torch_dtype("Float16") == torch.float16
    assert get_torch_dtype("BFLOAT16") == torch.bfloat16
    assert get_torch_dtype("AUTO") in [torch.float16, torch.bfloat16, torch.float32]


def test_storage_and_session_defaults(tmp_path):
    from app.config import load_config

    config = load_config(tmp_path / "missing.yaml")

    assert config.storage.data_dir == "./data"
    assert config.session.cookie_name == "transcriber_session"
    assert config.session.cookie_secure is False
    assert config.session.cookie_max_age_days == 365


def test_load_storage_and_session_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "storage:\n  data_dir: /srv/transcriber\n"
        "session:\n  cookie_secure: true\n  cookie_max_age_days: 30\n"
    )
    from app.config import load_config

    config = load_config(config_file)

    assert config.storage.data_dir == "/srv/transcriber"
    assert config.session.cookie_name == "transcriber_session"
    assert config.session.cookie_secure is True
    assert config.session.cookie_max_age_days == 30
