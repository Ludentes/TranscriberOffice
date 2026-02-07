import pytest
from unittest.mock import Mock, patch


def test_stop_flag_functions():
    """Verify stop flag get/set functions work."""
    from app.transcribe import set_stop_flag, check_stop_flag

    # Initially false
    set_stop_flag(False)
    assert not check_stop_flag()

    # Set to true
    set_stop_flag(True)
    assert check_stop_flag()

    # Reset to false
    set_stop_flag(False)
    assert not check_stop_flag()


def test_transcribe_stream_respects_stop_flag():
    """Verify transcribe_stream stops when flag is set."""
    from app.transcribe import TranscriptionService, set_stop_flag

    service = TranscriptionService()
    service._loaded = True

    # Set stop flag before starting
    set_stop_flag(True)

    with patch('librosa.get_duration', return_value=30.0):
        with patch.object(service, 'processor'), patch.object(service, 'model'):
            results = list(service.transcribe_stream("/fake/audio.mp3", None))

            # Should stop immediately
            assert any("stopped" in str(r[0]).lower() for r in results)
