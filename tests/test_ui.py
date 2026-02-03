# tests/test_ui.py
import pytest
from unittest.mock import Mock, patch


def test_create_gradio_interface():
    """Gradio interface creates without errors."""
    from app.transcribe import TranscriptionResult

    mock_service = Mock()
    mock_service.transcribe_stream.return_value = iter([
        ("Generating...", None),
        ("Final result", TranscriptionResult(
            success=True,
            segments=[],
            full_text="Test transcript",
            duration_seconds=10.0,
            speakers_detected=1,
            processing_time_seconds=5.0,
            error=None
        ))
    ])

    with patch('app.ui.get_transcription_service', return_value=mock_service):
        from app.ui import create_ui

        demo = create_ui()
        assert demo is not None


def test_process_audio_stream_function():
    """Process audio stream function yields expected results."""
    from app.transcribe import TranscriptionResult

    mock_service = Mock()
    mock_service.transcribe_stream.return_value = iter([
        ("--- Generating (1 tokens, 0.5s) ---\nHello", None),
        ('[Speaker 1] 00:00:00 - 00:00:05\n"Hello"\n', TranscriptionResult(
            success=True,
            segments=[{"speaker": "Speaker 1", "start": 0, "end": 5, "text": "Hello"}],
            full_text='[Speaker 1] 00:00:00 - 00:00:05\n"Hello"\n',
            duration_seconds=10.0,
            speakers_detected=1,
            processing_time_seconds=2.5,
            error=None
        ))
    ])

    with patch('app.ui.get_transcription_service', return_value=mock_service):
        from app.ui import process_audio_stream

        results = list(process_audio_stream("/fake/path.mp3", "hotword1, hotword2"))

        # Should have at least one result
        assert len(results) >= 1

        # Get the final result
        text_result, json_result = results[-1]

        assert "Speaker 1" in text_result
        assert "Hello" in text_result
        assert "Completed in" in text_result  # Processing time shown
        assert '"success": true' in json_result.lower() or "success" in json_result
        assert "processing_time_seconds" in json_result
