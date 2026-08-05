# tests/test_ui.py
import pytest
from unittest.mock import Mock, patch

from app.job_service import JobService
from app.job_store import JobStatus, JobStore


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


def test_history_choices_show_only_supplied_owner_jobs(tmp_path):
    from app.ui import build_history_choices

    store = JobStore(tmp_path / "jobs.sqlite3")
    alice = store.resolve_owner("6" * 64)
    bob = store.resolve_owner("7" * 64)
    store.create_job(alice, "alice.mp3", "/private/alice.mp3", "")
    store.create_job(bob, "bob.mp3", "/private/bob.mp3", "")

    choices = build_history_choices(store.list_jobs(alice))

    assert any("alice.mp3" in label for label, _ in choices)
    assert all("bob.mp3" not in label for label, _ in choices)


def test_download_links_are_available_only_for_completed_job(tmp_path):
    from app.ui import build_download_links

    store = JobStore(tmp_path / "jobs.sqlite3")
    owner = store.resolve_owner("8" * 64)
    job = store.create_job(owner, "meeting.mp3", "/private/meeting.mp3", "")
    assert build_download_links(job) == ""
    store.claim_next_job()
    store.complete_job(job.id, "text", "{}")

    completed = store.get_job(owner, job.id)
    markdown = build_download_links(completed)

    assert f"/api/jobs/{job.id}/download.txt" in markdown
    assert f"/api/jobs/{job.id}/download.json" in markdown


def test_default_history_selection_prefers_active_job(tmp_path):
    from app.ui import choose_history_job

    store = JobStore(tmp_path / "jobs.sqlite3")
    owner = store.resolve_owner("9" * 64)
    completed = store.create_job(owner, "old.mp3", "/private/old.mp3", "")
    store.claim_next_job()
    store.complete_job(completed.id, "text", "{}")
    active = store.create_job(owner, "active.mp3", "/private/active.mp3", "")

    assert choose_history_job(store.list_jobs(owner), None) == active.id


def test_create_persistent_gradio_interface(tmp_path):
    from app.ui import create_ui

    store = JobStore(tmp_path / "data" / "jobs.sqlite3")
    service = JobService(store, tmp_path / "data", max_file_size_mb=1)

    demo = create_ui(store, service)

    assert demo is not None
