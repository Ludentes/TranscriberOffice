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


def test_parse_model_output_current_json_format():
    """Current VibeVoice JSON output is parsed without its assistant prefix."""
    from app.transcribe import parse_model_output

    raw_output = '''assistant
[{"Start": 1.25, "End": 3.5, "Speaker": 2, "Content": "Привет."}]'''

    assert parse_model_output(raw_output) == [{
        "speaker_id": 2,
        "start_time": 1.25,
        "end_time": 3.5,
        "text": "Привет.",
    }]


def test_parse_model_output_rejects_malformed_raw_output():
    """Malformed generation must not become a fake 00:00 transcript segment."""
    from app.transcribe import parse_model_output

    assert parse_model_output("assistant\n[{Start0, broken output") == []


def test_salvage_complete_segments_removes_repeated_tail():
    from app.transcribe import salvage_complete_segments

    raw_output = (
        'assistant\n[{"Start":0,"End":2,"Speaker":0,"Content":"Готово."},'
        '{"Start":2,"End":4,"Speaker":0,"Content":"я, я, я, я, я, я, я'
    )

    segments = salvage_complete_segments(raw_output)

    assert [segment["text"] for segment in segments] == ["Готово."]
    assert "я, я" not in segments[0]["text"]


@pytest.mark.parametrize("text", [
    "я, " * 12,
    "ну, поэтому, " * 12,
    "длинный повтор " * 12,
])
def test_repetition_detector_detects_generation_loops(text):
    from app.transcribe import RepetitionDetector

    detector = RepetitionDetector()
    assert detector.add_text(text)


def test_repetition_detector_allows_natural_speech():
    from app.transcribe import RepetitionDetector

    detector = RepetitionDetector()
    text = (
        "Мы обсудили мобильное приложение, аналитику, публикации и затем "
        "перешли к плану работ на следующую неделю."
    )
    assert not detector.add_text(text)


def test_streaming_transcription_recovers_from_repetition(monkeypatch):
    """A looping greedy attempt is stopped and retried with sampling."""
    import queue
    import sys
    import threading
    import types
    from app.transcribe import TranscriptionService

    stop_signal = object()

    class FakeStreamer:
        def __init__(self, *args, **kwargs):
            self.text_queue = queue.Queue()
            self.stop_signal = stop_signal

        def __iter__(self):
            while True:
                item = self.text_queue.get(timeout=2)
                if item is self.stop_signal:
                    return
                yield item

    class FakeStoppingCriteria:
        pass

    transformers = types.ModuleType("transformers")
    transformers.TextIteratorStreamer = FakeStreamer
    transformers.StoppingCriteria = FakeStoppingCriteria
    transformers.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        types.SimpleNamespace(get_duration=lambda path: 30.0),
    )

    class FakeTensor:
        def to(self, device):
            return self

    class FakeTokenizer:
        eos_token_id = 1
        pad_token_id = 0

        @staticmethod
        def encode(text):
            return text

    class FakeProcessor:
        tokenizer = FakeTokenizer()
        pad_id = 0

        def __call__(self, **kwargs):
            return {"input_ids": FakeTensor()}

        @staticmethod
        def decode(value, **kwargs):
            return value

        @staticmethod
        def post_process_transcription(text):
            return [{
                "speaker_id": 0,
                "start_time": 0.0,
                "end_time": 1.0,
                "text": "Готово.",
            }]

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            streamer = kwargs["streamer"]
            if len(self.calls) == 1:
                streamer.text_queue.put("я, " * 12)
            else:
                streamer.text_queue.put(
                    'assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Готово."}]'
                )
            streamer.text_queue.put(streamer.stop_signal)

    service = TranscriptionService()
    service._loaded = True
    service.processor = FakeProcessor()
    service.model = FakeModel()

    results = list(
        service._transcribe_single_stream(
            "audio.wav", None, 100, stop_event=threading.Event()
        )
    )

    assert len(service.model.calls) == 2
    assert service.model.calls[0]["do_sample"] is False
    assert service.model.calls[1]["do_sample"] is True
    assert service.model.calls[1]["temperature"] == 0.2
    assert any("Retrying transcription" in message for message, _ in results)
    assert results[-1][1].success
    assert results[-1][1].full_text.endswith('"Готово."\n')


def test_looping_fragment_is_split_and_merged_without_markers(monkeypatch):
    """Recovery parts continue the job and only clean speech reaches the result."""
    import sys
    import types
    from app.transcribe import TranscriptionResult, TranscriptionService

    librosa = types.ModuleType("librosa")
    librosa.get_duration = lambda path: 120.0
    monkeypatch.setitem(sys.modules, "librosa", librosa)

    transcription_config = types.SimpleNamespace(
        silence_split=False,
        silence_noise_db=-30,
        silence_min_duration=0.5,
        silence_search_window=30,
    )
    monkeypatch.setattr(
        "app.transcribe.get_config",
        lambda: types.SimpleNamespace(transcription=transcription_config),
    )
    monkeypatch.setattr(
        "app.transcribe.split_audio",
        lambda *args, **kwargs: [("part-1.wav", 0.0), ("part-2.wav", 60.0)],
    )

    service = TranscriptionService()

    def transcribe_part(path, *args, **kwargs):
        text = "Первая часть." if path == "part-1.wav" else "Вторая часть."
        result = TranscriptionResult(
            success=True,
            segments=[{
                "speaker_id": 0,
                "start_time": 1.0,
                "end_time": 3.0,
                "text": text,
            }],
            full_text=text,
            duration_seconds=60.0,
            speakers_detected=1,
            error=None,
        )
        yield result.full_text, result

    monkeypatch.setattr(service, "_transcribe_single_stream", transcribe_part)

    outputs = list(
        service._recover_looping_audio(
            "source.wav", None, 100, None, recovery_depth=0
        )
    )
    result = outputs[-1][1]

    assert result.success
    assert [segment["start_time"] for segment in result.segments] == [1.0, 61.0]
    assert "Первая часть." in result.full_text
    assert "Вторая часть." in result.full_text
    assert "Recovery" not in result.full_text
    assert "repetition" not in result.full_text.lower()


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
