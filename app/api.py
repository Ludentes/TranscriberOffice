# app/api.py
"""REST API endpoints for transcription service."""
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_config
from app.transcribe import get_transcription_service
from app.transcription_queue import gpu_execution_lock, transcription_queue


router = APIRouter(prefix="/api", tags=["transcription"])


class TranscriptionSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

    @classmethod
    def from_raw(cls, seg: dict) -> "TranscriptionSegment":
        """Build from a pipeline segment, tolerating both naming conventions.

        The transcription pipeline may emit either start/end/speaker or
        start_time/end_time/speaker_id (see format_transcript), and speaker may
        be missing entirely. Normalize all of these here.
        """
        speaker_id = seg.get("speaker", seg.get("speaker_id", "Speaker 1"))
        if isinstance(speaker_id, (int, float)):
            speaker = f"Speaker {int(speaker_id)}"
        else:
            speaker = str(speaker_id)
        return cls(
            speaker=speaker,
            start=float(seg.get("start", seg.get("start_time", 0)) or 0),
            end=float(seg.get("end", seg.get("end_time", 0)) or 0),
            text=(seg.get("text") or "").strip(),
        )


class TranscriptionResponse(BaseModel):
    success: bool
    duration_seconds: float = 0
    speakers_detected: int = 0
    processing_time_seconds: float = 0
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    full_text: str = ""
    error: Optional[str] = None


@router.post("/transcribe", response_model=TranscriptionResponse)
def transcribe_audio(
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
    with transcription_queue.slot():
        content = file.file.read()
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
            with gpu_execution_lock:
                result = service.transcribe(
                    audio_path=tmp_path,
                    hotwords=hotwords
                )

            return TranscriptionResponse(
                success=result.success,
                duration_seconds=result.duration_seconds,
                speakers_detected=result.speakers_detected,
                processing_time_seconds=round(result.processing_time_seconds, 2),
                segments=[TranscriptionSegment.from_raw(seg) for seg in result.segments],
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
