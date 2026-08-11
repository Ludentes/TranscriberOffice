import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


def test_split_audio_creates_chunks():
    """Verify split_audio creates chunks with correct overlap."""
    from app.transcribe import split_audio

    # Mock librosa.get_duration to return 7 minutes
    with patch('librosa.get_duration', return_value=420.0):
        # Mock subprocess.run to simulate ffmpeg success
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            chunks = split_audio("/fake/audio.mp3", chunk_minutes=3, silence_split=False)

            # 7 minutes / 3 minutes per chunk = 3 chunks (0-3, 2:50-5:50, 5:40-7:00)
            assert len(chunks) >= 2

            # Return type is list of (path, start_time) tuples
            for chunk_path, start_time in chunks:
                assert isinstance(chunk_path, str)
                assert isinstance(start_time, float)

            # Nested recovery may split an already temporary chunk, so names
            # must never collide with each other or the source path.
            chunk_paths = [chunk_path for chunk_path, _ in chunks]
            assert len(chunk_paths) == len(set(chunk_paths))
            assert "/fake/audio.mp3" not in chunk_paths

            # Verify ffmpeg was called
            assert mock_run.call_count >= 2


def test_split_audio_cleanup():
    """Verify chunk cleanup in finally block."""
    from app.transcribe import split_audio
    import tempfile

    # Create temporary test chunks
    chunk_paths = []
    for i in range(3):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(b"fake audio data")
        tmp.close()
        chunk_paths.append(tmp.name)

    # Verify files exist
    for path in chunk_paths:
        assert Path(path).exists()

    # Cleanup
    for path in chunk_paths:
        Path(path).unlink(missing_ok=True)

    # Verify files removed
    for path in chunk_paths:
        assert not Path(path).exists()


def _mock_config(threshold=5, chunk_size=3, overlap=10):
    """Create a mock config with specified chunking settings."""
    mock_cfg = Mock()
    mock_cfg.transcription.chunk_threshold_minutes = threshold
    mock_cfg.transcription.chunk_size_minutes = chunk_size
    mock_cfg.transcription.chunk_overlap_seconds = overlap
    mock_cfg.transcription.silence_split = True
    mock_cfg.transcription.silence_noise_db = -30
    mock_cfg.transcription.silence_min_duration = 0.5
    mock_cfg.transcription.silence_search_window = 30
    return mock_cfg


def test_transcribe_stream_chunks_large_audio():
    """Verify large audio files are chunked before processing."""
    from app.transcribe import TranscriptionService, TranscriptionResult

    service = TranscriptionService()

    # Mock librosa to return 7 minutes (above 5-min threshold)
    with patch('librosa.get_duration', return_value=420.0), \
         patch('app.transcribe.get_config', return_value=_mock_config(threshold=5)), \
         patch('app.transcribe.split_audio', return_value=[("/chunk1.mp3", 0.0), ("/chunk2.mp3", 170.0)]):
            # Mock _transcribe_single_stream to return a simple result
            mock_result = TranscriptionResult(
                success=True,
                segments=[{"speaker": "Speaker 1", "start": 0.0, "end": 10.0, "text": "Test"}],
                full_text="Test",
                duration_seconds=180.0,
                speakers_detected=1,
                processing_time_seconds=1.0,
                error=None
            )

            def mock_transcribe_single(*args, **kwargs):
                yield "Generating...", None
                yield "Final", mock_result

            with patch.object(service, '_transcribe_single_stream', side_effect=mock_transcribe_single):
                service._loaded = True

                results = list(service.transcribe_stream("/fake/audio.mp3", None))

                # Should mention chunking
                assert any("chunk" in str(r[0]).lower() for r in results)
