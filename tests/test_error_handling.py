import pytest
from unittest.mock import Mock, patch


def test_error_messages_printed_to_console():
    """Verify errors are printed to console with traceback."""
    from app.transcribe import TranscriptionService
    import io
    import sys

    service = TranscriptionService()
    service._loaded = True

    # Mock to raise exception
    with patch('librosa.get_duration', side_effect=Exception("Test error")):
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            results = list(service.transcribe_stream("/fake/audio.mp3", None))

            # Should yield error message
            assert any("ERROR" in str(r[0]) for r in results)

        finally:
            sys.stdout = sys.__stdout__

            # Check console output
            output = captured_output.getvalue()
            assert "ERROR" in output or "error" in output.lower()


def test_progress_messages_during_chunking():
    """Verify progress messages are yielded during chunking."""
    from app.transcribe import TranscriptionService

    service = TranscriptionService()
    service._loaded = True

    with patch('librosa.get_duration', return_value=420.0):  # 7 minutes
        with patch('app.transcribe.split_audio', return_value=["/chunk1.mp3", "/chunk2.mp3"]):
            with patch.object(service, '_transcribe_single_stream', return_value=iter([
                ("result", Mock(segments=[], success=True))
            ])):
                results = list(service.transcribe_stream("/fake/audio.mp3", None))

                messages = [r[0] for r in results]

                # Should mention splitting
                assert any("splitting" in str(m).lower() for m in messages)

                # Should mention chunk progress
                assert any("chunk" in str(m).lower() for m in messages)
