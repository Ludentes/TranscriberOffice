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
