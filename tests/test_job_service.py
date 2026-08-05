from pathlib import Path

import pytest

from app.job_service import JobConflictError, JobService, JobValidationError
from app.job_store import JobStore


def make_service(tmp_path, max_mb=1):
    store = JobStore(tmp_path / "data" / "jobs.sqlite3")
    service = JobService(store, tmp_path / "data", max_file_size_mb=max_mb)
    owner = store.resolve_owner("4" * 64)
    return service, store, owner


def test_create_job_copies_audio_to_private_generated_path(tmp_path):
    service, store, owner = make_service(tmp_path)
    source = tmp_path / "My Meeting.mp3"
    source.write_bytes(b"audio")

    job = service.create_job(owner, source, source.name, "Project X")

    saved = Path(job.audio_path)
    assert saved.parent == tmp_path / "data" / "uploads"
    assert saved.name == f"{job.id}.mp3"
    assert saved.read_bytes() == b"audio"
    assert job.hotwords == "Project X"
    assert store.get_job(owner, job.id) == job


def test_create_job_rejects_extension_and_oversized_file(tmp_path):
    service, store, owner = make_service(tmp_path)
    wrong = tmp_path / "notes.txt"
    wrong.write_bytes(b"text")
    large = tmp_path / "large.mp3"
    large.write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(JobValidationError, match="Unsupported"):
        service.create_job(owner, wrong, wrong.name, "")
    with pytest.raises(JobValidationError, match="too large"):
        service.create_job(owner, large, large.name, "")

    assert store.list_jobs(owner) == []
    assert list(service.upload_dir.iterdir()) == []


def test_create_job_removes_file_when_database_write_fails(tmp_path, monkeypatch):
    service, store, owner = make_service(tmp_path)
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"audio")

    def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "create_job", fail)

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.create_job(owner, source, source.name, "")

    assert list(service.upload_dir.iterdir()) == []


def test_delete_requires_terminal_owned_job_and_cleans_audio(tmp_path):
    service, store, owner = make_service(tmp_path)
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"audio")
    job = service.create_job(owner, source, source.name, "")

    with pytest.raises(JobConflictError):
        service.delete_job(owner, job.id)

    store.claim_next_job()
    store.fail_job(job.id, "failed")
    assert service.delete_job(owner, job.id) is True
    assert store.get_job(owner, job.id) is None
    assert not Path(job.audio_path).exists()


def test_download_name_is_sanitized(tmp_path):
    service, _, _ = make_service(tmp_path)

    assert service.download_name("../Quarter report?.mp3", "txt") == "Quarter_report_transcript.txt"
    assert service.download_name("meeting.mp3", "json") == "meeting_transcript.json"
