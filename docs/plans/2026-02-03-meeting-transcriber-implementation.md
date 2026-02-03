# Meeting Transcriber Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web app and API for transcribing meetings using VibeVoice-ASR with Gradio UI and FastAPI backend.

**Architecture:** FastAPI serves both a Gradio web interface (for browser uploads) and a REST API endpoint (for n8n). A shared transcription service wraps the VibeVoice-ASR model loaded once at startup.

**Tech Stack:** Python 3.10+, FastAPI, Gradio, VibeVoice-ASR (HuggingFace), PyTorch with CUDA, FFmpeg

---

## Task 1: Project Structure and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `config.yaml`
- Create: `.gitignore`

**Step 1: Create requirements.txt**

```txt
# Core
torch>=2.0.0
transformers>=4.51.3,<5.0.0
accelerate
safetensors

# VibeVoice dependencies
librosa
scipy
numpy
numba>0.57.0
llvmlite>0.40.0
pydub
av

# Web framework
fastapi
uvicorn[standard]
gradio>=4.0.0
python-multipart

# Configuration
pyyaml

# Audio processing
ffmpeg-python
```

**Step 2: Create config.yaml**

```yaml
server:
  host: "0.0.0.0"
  port: 7860

model:
  # Options: "microsoft/VibeVoice-ASR" or "scerz/VibeVoice-ASR-4bit"
  path: "microsoft/VibeVoice-ASR"
  # Options: "auto", "bfloat16", "float16", "float32"
  dtype: "auto"
  # Local cache directory
  cache_dir: "./models"
  # Attention implementation: "sdpa", "eager" (flash_attention_2 requires separate install)
  attn_implementation: "sdpa"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
  default_max_new_tokens: 8192
```

**Step 3: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project
models/
*.mp3
*.wav
*.m4a
uploads/
*.log

# OS
.DS_Store
Thumbs.db
```

**Step 4: Create app/__init__.py**

```python
"""Meeting Transcriber - VibeVoice-ASR based transcription service."""
```

**Step 5: Commit**

```bash
git add requirements.txt config.yaml .gitignore app/__init__.py
git commit -m "chore: add project structure and dependencies"
```

---

## Task 2: Configuration Loader

**Files:**
- Create: `app/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Create tests/__init__.py**

```python
"""Tests for Meeting Transcriber."""
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.config'"

**Step 4: Write minimal implementation**

```python
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
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add configuration loader with auto dtype detection"
```

---

## Task 3: Transcription Service

**Files:**
- Create: `app/transcribe.py`
- Create: `tests/test_transcribe.py`

**Step 1: Write the failing test**

```python
# tests/test_transcribe.py
import pytest
from dataclasses import dataclass


def test_format_transcript_segments():
    """Transcript segments format correctly to human-readable text."""
    from app.transcribe import format_transcript

    segments = [
        {"speaker": "Speaker 1", "start": 5.0, "end": 12.0, "text": "Hello everyone."},
        {"speaker": "Speaker 2", "start": 13.0, "end": 25.0, "text": "Good morning."},
    ]

    result = format_transcript(segments)

    assert "[Speaker 1] 00:00:05 - 00:00:12" in result
    assert '"Hello everyone."' in result
    assert "[Speaker 2] 00:00:13 - 00:00:25" in result


def test_format_timestamp():
    """Timestamps format as HH:MM:SS."""
    from app.transcribe import format_timestamp

    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(5.5) == "00:00:05"
    assert format_timestamp(65) == "00:01:05"
    assert format_timestamp(3661) == "01:01:01"


def test_parse_model_output():
    """Model output parses into structured segments."""
    from app.transcribe import parse_model_output

    # VibeVoice outputs text in a specific format - test the parser
    raw_output = """<|speaker_1|> Hello everyone, let's begin. <|end|>
<|speaker_2|> Sure, I have updates. <|end|>"""

    segments = parse_model_output(raw_output)

    assert len(segments) >= 1
    assert "speaker" in segments[0]
    assert "text" in segments[0]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.transcribe'"

**Step 3: Write minimal implementation**

```python
# app/transcribe.py
"""Transcription service using VibeVoice-ASR."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class TranscriptionResult:
    """Result from transcription."""
    success: bool
    segments: list[dict]
    full_text: str
    duration_seconds: float
    speakers_detected: int
    error: Optional[str] = None


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript(segments: list[dict]) -> str:
    """Format segments into human-readable transcript."""
    lines = []
    for seg in segments:
        start = format_timestamp(seg.get("start", 0))
        end = format_timestamp(seg.get("end", 0))
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        lines.append(f'[{speaker}] {start} - {end}')
        lines.append(f'"{text}"')
        lines.append("")
    return "\n".join(lines)


def parse_model_output(raw_output: str) -> list[dict]:
    """Parse VibeVoice model output into structured segments.

    VibeVoice outputs in format like:
    <|speaker_1|> text <|end|>
    Or with timestamps:
    <|0.00|> <|speaker_1|> text <|end|>
    """
    segments = []

    # Pattern to match speaker segments with optional timestamps
    # This handles various VibeVoice output formats
    pattern = r'(?:<\|(\d+\.?\d*)\|>\s*)?<\|speaker_(\d+)\|>\s*(.*?)\s*<\|end\|>'

    matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)

    current_time = 0.0
    for match in matches:
        timestamp_str, speaker_id, text = match

        start_time = float(timestamp_str) if timestamp_str else current_time
        # Estimate end time based on text length (rough: 150 words/min)
        words = len(text.split())
        duration = max(1.0, words / 2.5)  # ~150 wpm
        end_time = start_time + duration
        current_time = end_time

        segments.append({
            "speaker": f"Speaker {speaker_id}",
            "start": start_time,
            "end": end_time,
            "text": text.strip()
        })

    # If no pattern matches, try simpler parsing or return raw
    if not segments and raw_output.strip():
        segments.append({
            "speaker": "Speaker 1",
            "start": 0.0,
            "end": 0.0,
            "text": raw_output.strip()
        })

    return segments


class TranscriptionService:
    """Service for transcribing audio using VibeVoice-ASR."""

    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-ASR",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        attn_implementation: str = "sdpa"
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.cache_dir = cache_dir
        self.attn_implementation = attn_implementation
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the VibeVoice-ASR model."""
        if self._loaded:
            return

        # Import here to avoid loading at module import time
        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}, dtype: {self.dtype}")

        # Try VibeVoice-specific imports first, fall back to auto
        try:
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
            from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

            self.processor = VibeVoiceASRProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir,
                attn_implementation=self.attn_implementation,
                trust_remote_code=True
            )
        except ImportError:
            # Fallback to AutoModel with trust_remote_code
            print("VibeVoice package not found, using AutoModel with trust_remote_code")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir,
                attn_implementation=self.attn_implementation,
                trust_remote_code=True
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print("Model loaded successfully")

    def transcribe(
        self,
        audio_path: str,
        hotwords: Optional[str] = None,
        max_new_tokens: int = 8192
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            hotwords: Optional comma-separated hotwords for better recognition
            max_new_tokens: Maximum tokens to generate

        Returns:
            TranscriptionResult with segments and formatted text
        """
        if not self._loaded:
            self.load_model()

        try:
            # Build context info from hotwords
            context_info = None
            if hotwords:
                terms = [h.strip() for h in hotwords.split(",") if h.strip()]
                if terms:
                    context_info = f"Key terms: {', '.join(terms)}"

            # Process audio
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                add_generation_prompt=True,
                context_info=context_info
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode output
            generated_text = self.processor.decode(
                output_ids[0],
                skip_special_tokens=False  # Keep speaker tokens for parsing
            )

            # Try to use processor's post_process if available
            try:
                segments = self.processor.post_process_transcription(generated_text)
                # Convert to our format if needed
                if segments and isinstance(segments[0], dict):
                    pass  # Already in dict format
                else:
                    segments = parse_model_output(generated_text)
            except (AttributeError, Exception):
                segments = parse_model_output(generated_text)

            # Get audio duration
            import librosa
            duration = librosa.get_duration(path=audio_path)

            # Count unique speakers
            speakers = set(seg.get("speaker", "") for seg in segments)

            return TranscriptionResult(
                success=True,
                segments=segments,
                full_text=format_transcript(segments),
                duration_seconds=duration,
                speakers_detected=len(speakers),
                error=None
            )

        except Exception as e:
            return TranscriptionResult(
                success=False,
                segments=[],
                full_text="",
                duration_seconds=0,
                speakers_detected=0,
                error=str(e)
            )


# Global service instance
_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get or create the global transcription service."""
    global _service
    if _service is None:
        from app.config import get_config, get_torch_dtype
        config = get_config()
        _service = TranscriptionService(
            model_path=config.model.path,
            dtype=get_torch_dtype(config.model.dtype),
            cache_dir=config.model.cache_dir,
            attn_implementation=config.model.attn_implementation
        )
    return _service
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat: add transcription service with VibeVoice-ASR"
```

---

## Task 4: REST API Endpoint

**Files:**
- Create: `app/api.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing test**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def mock_transcription_service():
    """Mock the transcription service for API tests."""
    from app.transcribe import TranscriptionResult

    mock_service = Mock()
    mock_service.transcribe.return_value = TranscriptionResult(
        success=True,
        segments=[
            {"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "Hello world."}
        ],
        full_text='[Speaker 1] 00:00:00 - 00:00:05\n"Hello world."\n',
        duration_seconds=10.0,
        speakers_detected=1,
        error=None
    )
    return mock_service


@pytest.fixture
def client(mock_transcription_service):
    """Create test client with mocked service."""
    with patch('app.api.get_transcription_service', return_value=mock_transcription_service):
        from app.api import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        yield TestClient(app)


def test_transcribe_endpoint_success(client, tmp_path, mock_transcription_service):
    """POST /api/transcribe returns JSON transcript."""
    # Create a dummy audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio content")

    with open(audio_file, "rb") as f:
        response = client.post(
            "/api/transcribe",
            files={"file": ("test.mp3", f, "audio/mpeg")},
            data={"hotwords": "ProjectX, John"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "segments" in data
    assert "full_text" in data
    assert data["speakers_detected"] == 1


def test_transcribe_endpoint_no_file(client):
    """POST /api/transcribe without file returns error."""
    response = client.post("/api/transcribe")
    assert response.status_code == 422  # Validation error
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.api'"

**Step 3: Write minimal implementation**

```python
# app/api.py
"""REST API endpoints for transcription service."""
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.transcribe import get_transcription_service


router = APIRouter(prefix="/api", tags=["transcription"])


class TranscriptionSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    success: bool
    duration_seconds: float = 0
    speakers_detected: int = 0
    segments: list[TranscriptionSegment] = []
    full_text: str = ""
    error: Optional[str] = None


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe (MP3)"),
    hotwords: Optional[str] = Form(None, description="Comma-separated hotwords")
) -> TranscriptionResponse:
    """Transcribe an audio file.

    Upload an MP3 file and receive a structured transcript with
    speaker identification and timestamps.

    Args:
        file: Audio file (MP3 format)
        hotwords: Optional comma-separated terms for better recognition

    Returns:
        JSON with success status, segments, and formatted transcript
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return TranscriptionResponse(
            success=False,
            error=f"File format not supported. Expected one of: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file to temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
        service = get_transcription_service()
        result = service.transcribe(
            audio_path=tmp_path,
            hotwords=hotwords
        )

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        return TranscriptionResponse(
            success=result.success,
            duration_seconds=result.duration_seconds,
            speakers_detected=result.speakers_detected,
            segments=[TranscriptionSegment(**seg) for seg in result.segments],
            full_text=result.full_text,
            error=result.error
        )

    except Exception as e:
        return TranscriptionResponse(
            success=False,
            error=f"Transcription failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api.py tests/test_api.py
git commit -m "feat: add REST API endpoint for transcription"
```

---

## Task 5: Gradio UI

**Files:**
- Create: `app/ui.py`
- Create: `tests/test_ui.py`

**Step 1: Write the failing test**

```python
# tests/test_ui.py
import pytest
from unittest.mock import Mock, patch


def test_create_gradio_interface():
    """Gradio interface creates without errors."""
    from app.transcribe import TranscriptionResult

    mock_service = Mock()
    mock_service.transcribe.return_value = TranscriptionResult(
        success=True,
        segments=[],
        full_text="Test transcript",
        duration_seconds=10.0,
        speakers_detected=1,
        error=None
    )

    with patch('app.ui.get_transcription_service', return_value=mock_service):
        from app.ui import create_ui

        demo = create_ui()
        assert demo is not None


def test_process_audio_function():
    """Process audio function returns expected tuple."""
    from app.transcribe import TranscriptionResult

    mock_service = Mock()
    mock_service.transcribe.return_value = TranscriptionResult(
        success=True,
        segments=[{"speaker": "Speaker 1", "start": 0, "end": 5, "text": "Hello"}],
        full_text='[Speaker 1] 00:00:00 - 00:00:05\n"Hello"\n',
        duration_seconds=10.0,
        speakers_detected=1,
        error=None
    )

    with patch('app.ui.get_transcription_service', return_value=mock_service):
        from app.ui import process_audio

        text_result, json_result = process_audio("/fake/path.mp3", "hotword1, hotword2")

        assert "Speaker 1" in text_result
        assert "Hello" in text_result
        assert '"success": true' in json_result.lower() or "success" in json_result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.ui'"

**Step 3: Write minimal implementation**

```python
# app/ui.py
"""Gradio web interface for transcription."""
import json
from typing import Optional

import gradio as gr

from app.transcribe import get_transcription_service


def process_audio(audio_path: Optional[str], hotwords: str) -> tuple[str, str]:
    """Process uploaded audio and return transcript.

    Args:
        audio_path: Path to uploaded audio file
        hotwords: Comma-separated hotwords

    Returns:
        Tuple of (formatted_text, json_string)
    """
    if not audio_path:
        return "Please upload an audio file.", "{}"

    service = get_transcription_service()
    result = service.transcribe(
        audio_path=audio_path,
        hotwords=hotwords if hotwords else None
    )

    if not result.success:
        error_msg = f"Transcription failed: {result.error}"
        error_json = json.dumps({"success": False, "error": result.error}, indent=2)
        return error_msg, error_json

    # Build JSON response
    json_response = {
        "success": True,
        "duration_seconds": result.duration_seconds,
        "speakers_detected": result.speakers_detected,
        "segments": result.segments,
        "full_text": result.full_text
    }

    return result.full_text, json.dumps(json_response, indent=2)


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Meeting Transcriber",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# Meeting Transcriber")
        gr.Markdown("Upload an MP3 file to transcribe with speaker identification and timestamps.")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    sources=["upload"],
                )

                hotwords_input = gr.Textbox(
                    label="Hotwords (optional)",
                    placeholder="ProjectX, John Smith, Q4 OKRs",
                    info="Comma-separated terms to improve recognition"
                )

                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Transcript"):
                    text_output = gr.Textbox(
                        label="Transcription",
                        lines=20,
                        show_copy_button=True
                    )

                with gr.Tab("JSON"):
                    json_output = gr.Code(
                        label="JSON Output",
                        language="json",
                        lines=20
                    )

        with gr.Row():
            gr.Markdown(
                "**Tip:** For best results, ensure clear audio quality. "
                "Add relevant names and terms as hotwords."
            )

        # Connect the button
        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, hotwords_input],
            outputs=[text_output, json_output]
        )

    return demo
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/ui.py tests/test_ui.py
git commit -m "feat: add Gradio web interface"
```

---

## Task 6: Main Application Entry Point

**Files:**
- Create: `app/main.py`
- Create: `tests/test_main.py`

**Step 1: Write the failing test**

```python
# tests/test_main.py
import pytest
from unittest.mock import Mock, patch


def test_create_app():
    """FastAPI app creates with routes mounted."""
    with patch('app.main.create_ui') as mock_ui:
        mock_ui.return_value = Mock()

        from app.main import create_app

        app = create_app()

        # Check API routes are registered
        routes = [route.path for route in app.routes]
        assert "/api/transcribe" in routes or any("/api" in r for r in routes)
        assert "/api/health" in routes or any("health" in r for r in routes)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.main'"

**Step 3: Write minimal implementation**

```python
# app/main.py
"""Main application entry point."""
import gradio as gr
from fastapi import FastAPI

from app.api import router as api_router
from app.config import get_config
from app.ui import create_ui


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Meeting Transcriber",
        description="Transcribe meetings with speaker identification using VibeVoice-ASR",
        version="1.0.0"
    )

    # Include API routes
    app.include_router(api_router)

    # Mount Gradio UI
    ui = create_ui()
    app = gr.mount_gradio_app(app, ui, path="/")

    return app


def main():
    """Run the application."""
    import uvicorn

    config = get_config()

    print(f"Starting Meeting Transcriber on {config.server.host}:{config.server.port}")
    print(f"Web UI: http://{config.server.host}:{config.server.port}/")
    print(f"API: http://{config.server.host}:{config.server.port}/api/transcribe")

    uvicorn.run(
        create_app(),
        host=config.server.host,
        port=config.server.port
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add main application entry point with Gradio mount"
```

---

## Task 7: Linux Install Script

**Files:**
- Create: `install.sh`

**Step 1: Write the install script**

```bash
#!/bin/bash
# install.sh - Meeting Transcriber installation script for Linux

set -e

echo "=========================================="
echo "  Meeting Transcriber - Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root is not recommended.${NC}"
fi

# Check Python version
echo ""
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        echo -e "${RED}Error: Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check NVIDIA GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓ Found GPU: $GPU_NAME ($GPU_MEMORY)${NC}"

    # Detect GPU architecture for dtype recommendation
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    if [ "$COMPUTE_CAP" -ge 80 ]; then
        RECOMMENDED_DTYPE="bfloat16"
        RECOMMENDED_MODEL="microsoft/VibeVoice-ASR"
        echo "  Ampere+ GPU detected - will use BF16"
    else
        RECOMMENDED_DTYPE="float16"
        RECOMMENDED_MODEL="scerz/VibeVoice-ASR-4bit"
        echo "  Pre-Ampere GPU detected - will use FP16 with 4-bit model"
    fi
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU support may not work.${NC}"
    RECOMMENDED_DTYPE="float32"
    RECOMMENDED_MODEL="scerz/VibeVoice-ASR-4bit"
fi

# Check FFmpeg
echo ""
echo "Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ FFmpeg $FFMPEG_VERSION${NC}"
else
    echo -e "${YELLOW}FFmpeg not found. Attempting to install...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    else
        echo -e "${RED}Error: Could not install FFmpeg. Please install manually.${NC}"
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Recreating...${NC}"
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Clone and install VibeVoice
echo ""
echo "Installing VibeVoice..."
if [ ! -d "VibeVoice" ]; then
    git clone https://github.com/microsoft/VibeVoice.git
fi
cd VibeVoice
pip install -e .
cd ..

# Create config if not exists
echo ""
echo "Creating configuration..."
if [ ! -f "config.yaml" ]; then
    cat > config.yaml << EOF
server:
  host: "0.0.0.0"
  port: 7860

model:
  path: "$RECOMMENDED_MODEL"
  dtype: "$RECOMMENDED_DTYPE"
  cache_dir: "./models"
  attn_implementation: "sdpa"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
  default_max_new_tokens: 8192
EOF
    echo -e "${GREEN}✓ Created config.yaml with recommended settings${NC}"
else
    echo -e "${YELLOW}config.yaml already exists, skipping${NC}"
fi

# Create models directory
mkdir -p models

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=========================================="
echo -e "${GREEN}  Installation complete!${NC}"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python -m app.main"
echo ""
```

**Step 2: Make executable and commit**

```bash
chmod +x install.sh
git add install.sh
git commit -m "feat: add Linux install script"
```

---

## Task 8: Linux Run Script

**Files:**
- Create: `run.sh`

**Step 1: Write the run script**

```bash
#!/bin/bash
# run.sh - Start the Meeting Transcriber server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run ./install.sh first."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Start server
echo "Starting Meeting Transcriber..."
python -m app.main "$@"
```

**Step 2: Make executable and commit**

```bash
chmod +x run.sh
git add run.sh
git commit -m "feat: add Linux run script"
```

---

## Task 9: Windows Install Script

**Files:**
- Create: `install.ps1`

**Step 1: Write the install script**

```powershell
# install.ps1 - Meeting Transcriber installation script for Windows
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Meeting Transcriber - Installation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Python version
Write-Host ""
Write-Host "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    $versionMatch = $pythonVersion -match 'Python (\d+)\.(\d+)'
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "Error: Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
            exit 1
        }
        Write-Host "✓ $pythonVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "Error: Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check NVIDIA GPU
Write-Host ""
Write-Host "Checking GPU..."
$recommendedDtype = "float32"
$recommendedModel = "scerz/VibeVoice-ASR-4bit"

try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Found GPU: $gpuInfo" -ForegroundColor Green

        # Parse compute capability
        if ($gpuInfo -match '(\d+)\.(\d+)$') {
            $computeMajor = [int]$Matches[1]
            if ($computeMajor -ge 8) {
                $recommendedDtype = "bfloat16"
                $recommendedModel = "microsoft/VibeVoice-ASR"
                Write-Host "  Ampere+ GPU detected - will use BF16"
            } else {
                $recommendedDtype = "float16"
                Write-Host "  Pre-Ampere GPU detected - will use FP16 with 4-bit model"
            }
        }
    }
} catch {
    Write-Host "Warning: nvidia-smi not found. GPU support may not work." -ForegroundColor Yellow
}

# Check FFmpeg
Write-Host ""
Write-Host "Checking FFmpeg..."
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "✓ FFmpeg found" -ForegroundColor Green
} catch {
    Write-Host "Warning: FFmpeg not found. Please install from https://ffmpeg.org/download.html" -ForegroundColor Yellow
    Write-Host "Add FFmpeg to your PATH after installation." -ForegroundColor Yellow
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..."
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Recreating..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}
python -m venv venv
.\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment created" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
Write-Host ""
Write-Host "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
Write-Host ""
Write-Host "Installing dependencies..."
pip install -r requirements.txt

# Clone and install VibeVoice
Write-Host ""
Write-Host "Installing VibeVoice..."
if (-not (Test-Path "VibeVoice")) {
    git clone https://github.com/microsoft/VibeVoice.git
}
Push-Location VibeVoice
pip install -e .
Pop-Location

# Create config if not exists
Write-Host ""
Write-Host "Creating configuration..."
if (-not (Test-Path "config.yaml")) {
    @"
server:
  host: "0.0.0.0"
  port: 7860

model:
  path: "$recommendedModel"
  dtype: "$recommendedDtype"
  cache_dir: "./models"
  attn_implementation: "sdpa"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
  default_max_new_tokens: 8192
"@ | Out-File -FilePath "config.yaml" -Encoding UTF8
    Write-Host "✓ Created config.yaml with recommended settings" -ForegroundColor Green
} else {
    Write-Host "config.yaml already exists, skipping" -ForegroundColor Yellow
}

# Create models directory
New-Item -ItemType Directory -Force -Path "models" | Out-Null

# Verify installation
Write-Host ""
Write-Host "Verifying installation..."
python -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"@

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the server:"
Write-Host "  .\run.ps1"
Write-Host ""
Write-Host "Or manually:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python -m app.main"
Write-Host ""
```

**Step 2: Commit**

```bash
git add install.ps1
git commit -m "feat: add Windows install script"
```

---

## Task 10: Windows Run Script

**Files:**
- Create: `run.ps1`

**Step 1: Write the run script**

```powershell
# run.ps1 - Start the Meeting Transcriber server
# Run with: powershell -ExecutionPolicy Bypass -File run.ps1

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "Error: Virtual environment not found. Run install.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate venv
.\venv\Scripts\Activate.ps1

# Start server
Write-Host "Starting Meeting Transcriber..."
python -m app.main $args
```

**Step 2: Commit**

```bash
git add run.ps1
git commit -m "feat: add Windows run script"
```

---

## Task 11: Run All Tests

**Step 1: Run complete test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Final commit**

```bash
git add -A
git commit -m "chore: finalize MVP implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project structure | requirements.txt, config.yaml, .gitignore, app/__init__.py |
| 2 | Configuration loader | app/config.py, tests/test_config.py |
| 3 | Transcription service | app/transcribe.py, tests/test_transcribe.py |
| 4 | REST API | app/api.py, tests/test_api.py |
| 5 | Gradio UI | app/ui.py, tests/test_ui.py |
| 6 | Main entry point | app/main.py, tests/test_main.py |
| 7 | Linux install | install.sh |
| 8 | Linux run | run.sh |
| 9 | Windows install | install.ps1 |
| 10 | Windows run | run.ps1 |
| 11 | Final verification | Run all tests |

**After implementation:** Test on target hardware with real audio file.
