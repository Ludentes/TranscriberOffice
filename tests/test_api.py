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
