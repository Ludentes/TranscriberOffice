# app/api.py
"""REST API endpoints for transcription service."""
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import get_config
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

    # Read file content and check size
    content = await file.read()
    config = get_config()
    max_size_bytes = config.transcription.max_file_size_mb * 1024 * 1024
    if len(content) > max_size_bytes:
        return TranscriptionResponse(
            success=False,
            error=f"File too large. Maximum size is {config.transcription.max_file_size_mb}MB"
        )

    # Save uploaded file to temp location
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
        service = get_transcription_service()
        result = service.transcribe(
            audio_path=tmp_path,
            hotwords=hotwords
        )

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
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
