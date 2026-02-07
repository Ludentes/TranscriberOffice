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

    assert len(segments) == 2
    assert "speaker" in segments[0]
    assert "text" in segments[0]


def test_vibevoice_import_error_message():
    """Verify clear error message when VibeVoice not installed."""
    import builtins
    from unittest.mock import patch
    from app.transcribe import TranscriptionService

    service = TranscriptionService()

    # Mock the vibevoice module imports to fail
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if 'vibevoice' in name:
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        with pytest.raises(ImportError) as exc_info:
            service.load_model()

        assert "VibeVoice package not installed" in str(exc_info.value)
        assert "./install.sh" in str(exc_info.value)


def test_model_switching_unloads_old_model():
    """Verify old model is unloaded when switching models."""
    import sys
    import torch
    from unittest.mock import MagicMock, patch
    from app.transcribe import TranscriptionService

    # Mock the vibevoice module before importing
    mock_vibevoice = MagicMock()
    mock_processor_class = MagicMock()
    mock_model_class = MagicMock()

    mock_vibevoice.processor.vibevoice_asr_processor.VibeVoiceASRProcessor = mock_processor_class
    mock_vibevoice.modular.modeling_vibevoice_asr.VibeVoiceASRForConditionalGeneration = mock_model_class

    sys.modules['vibevoice'] = mock_vibevoice
    sys.modules['vibevoice.processor'] = mock_vibevoice.processor
    sys.modules['vibevoice.processor.vibevoice_asr_processor'] = mock_vibevoice.processor.vibevoice_asr_processor
    sys.modules['vibevoice.modular'] = mock_vibevoice.modular
    sys.modules['vibevoice.modular.modeling_vibevoice_asr'] = mock_vibevoice.modular.modeling_vibevoice_asr

    try:
        # Setup mocks
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        first_mock_model = MagicMock()
        second_mock_model = MagicMock()
        mock_model_class.from_pretrained.side_effect = [first_mock_model, second_mock_model]

        # Create service with initial model
        service = TranscriptionService(model_path="microsoft/VibeVoice-ASR")
        service.load_model()

        assert service._loaded
        assert service.current_model_path == "microsoft/VibeVoice-ASR"
        first_model = service.model

        # Switch to different model path
        service.model_path = "scerz/VibeVoice-ASR-4bit"
        service.load_model()

        # Verify old model was unloaded and new one loaded
        assert service._loaded
        assert service.current_model_path == "scerz/VibeVoice-ASR-4bit"
        assert service.model is not first_model
    finally:
        # Cleanup mocked modules
        for mod in ['vibevoice', 'vibevoice.processor', 'vibevoice.processor.vibevoice_asr_processor',
                    'vibevoice.modular', 'vibevoice.modular.modeling_vibevoice_asr']:
            if mod in sys.modules:
                del sys.modules[mod]


def test_unload_model_clears_memory():
    """Verify unload_model frees resources properly."""
    import sys
    from unittest.mock import MagicMock
    from app.transcribe import TranscriptionService

    # Mock the vibevoice module before importing
    mock_vibevoice = MagicMock()
    mock_processor_class = MagicMock()
    mock_model_class = MagicMock()

    mock_vibevoice.processor.vibevoice_asr_processor.VibeVoiceASRProcessor = mock_processor_class
    mock_vibevoice.modular.modeling_vibevoice_asr.VibeVoiceASRForConditionalGeneration = mock_model_class

    sys.modules['vibevoice'] = mock_vibevoice
    sys.modules['vibevoice.processor'] = mock_vibevoice.processor
    sys.modules['vibevoice.processor.vibevoice_asr_processor'] = mock_vibevoice.processor.vibevoice_asr_processor
    sys.modules['vibevoice.modular'] = mock_vibevoice.modular
    sys.modules['vibevoice.modular.modeling_vibevoice_asr'] = mock_vibevoice.modular.modeling_vibevoice_asr

    try:
        # Setup mocks
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        service = TranscriptionService()
        service.load_model()

        assert service.model is not None
        assert service.processor is not None
        assert service._loaded

        service.unload_model()

        assert service.model is None
        assert service.processor is None
        assert not service._loaded
    finally:
        # Cleanup mocked modules
        for mod in ['vibevoice', 'vibevoice.processor', 'vibevoice.processor.vibevoice_asr_processor',
                    'vibevoice.modular', 'vibevoice.modular.modeling_vibevoice_asr']:
            if mod in sys.modules:
                del sys.modules[mod]
