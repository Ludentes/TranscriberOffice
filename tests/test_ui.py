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
