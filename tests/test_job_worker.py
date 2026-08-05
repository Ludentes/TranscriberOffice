import json
from pathlib import Path

from app.job_service import JobService
from app.job_store import JobStatus, JobStore
from app.job_worker import JobWorker
from app.transcribe import TranscriptionResult


class SuccessfulService:
    def transcribe_stream(self, audio_path, hotwords, stop_event):
        yield "halfway", None
        yield "final transcript", TranscriptionResult(
            success=True,
            segments=[{"speaker": "Speaker 1", "start": 0, "end": 1, "text": "Hi"}],
            full_text="final transcript",
            duration_seconds=1,
            speakers_detected=1,
            processing_time_seconds=0.5,
        )


class FailingService:
    def transcribe_stream(self, audio_path, hotwords, stop_event):
        raise RuntimeError("private filesystem detail")
        yield


class UnsafeResultService:
    def transcribe_stream(self, audio_path, hotwords, stop_event):
        yield "failed", TranscriptionResult(
            success=False,
            segments=[],
            full_text="",
            duration_seconds=0,
            speakers_detected=0,
            error="Decoder failed for /private/customer/meeting.mp3",
        )


def queued_job(tmp_path, name="meeting.mp3"):
    data_dir = tmp_path / "data"
    store = JobStore(data_dir / "jobs.sqlite3")
    service = JobService(store, data_dir, max_file_size_mb=1)
    owner = store.resolve_owner("5" * 64)
    source = tmp_path / name
    source.write_bytes(b"audio")
    job = service.create_job(owner, source, name, "names")
    return store, service, owner, job


def test_worker_persists_result_and_removes_audio(tmp_path):
    store, job_service, owner, job = queued_job(tmp_path)
    worker = JobWorker(store, job_service, lambda: SuccessfulService(), poll_seconds=0.01)

    assert worker.run_once() is True

    saved = store.get_job(owner, job.id)
    assert saved.status == JobStatus.COMPLETED
    assert saved.result_text == "final transcript"
    assert json.loads(saved.result_json)["success"] is True
    assert saved.audio_path is None
    assert not Path(job.audio_path).exists()


def test_worker_records_safe_failure_and_can_process_next_job(tmp_path):
    store, job_service, owner, first = queued_job(tmp_path, "first.mp3")
    source = tmp_path / "second.mp3"
    source.write_bytes(b"audio")
    second = job_service.create_job(owner, source, source.name, "")
    services = iter([FailingService(), SuccessfulService()])
    worker = JobWorker(store, job_service, lambda: next(services), poll_seconds=0.01)

    assert worker.run_once() is True
    failed = store.get_job(owner, first.id)
    assert failed.status == JobStatus.FAILED
    assert failed.error_message == "Transcription failed. Check server logs for details."
    assert "filesystem" not in failed.error_message

    assert worker.run_once() is True
    assert store.get_job(owner, second.id).status == JobStatus.COMPLETED


def test_worker_does_not_store_model_error_details_in_history(tmp_path):
    store, job_service, owner, job = queued_job(tmp_path)
    worker = JobWorker(store, job_service, lambda: UnsafeResultService(), poll_seconds=0.01)

    assert worker.run_once() is True

    failed = store.get_job(owner, job.id)
    assert failed.status == JobStatus.FAILED
    assert failed.error_message == "Transcription failed. Check server logs for details."
    assert "/private/" not in failed.error_message


def test_worker_marks_missing_audio_failed(tmp_path):
    store, job_service, owner, job = queued_job(tmp_path)
    Path(job.audio_path).unlink()
    worker = JobWorker(store, job_service, lambda: SuccessfulService(), poll_seconds=0.01)

    assert worker.run_once() is True

    saved = store.get_job(owner, job.id)
    assert saved.status == JobStatus.FAILED
    assert saved.error_message == "Source audio is unavailable. Please upload it again."


def test_worker_honors_running_cancellation(tmp_path):
    store, job_service, owner, job = queued_job(tmp_path)

    class CancelingService:
        def transcribe_stream(self, audio_path, hotwords, stop_event):
            store.request_cancel(owner, job.id)
            yield "working", None
            assert stop_event.is_set()
            yield "stopped", TranscriptionResult(
                success=False,
                segments=[],
                full_text="",
                duration_seconds=0,
                speakers_detected=0,
                error="Stopped",
            )

    worker = JobWorker(store, job_service, lambda: CancelingService(), poll_seconds=0.01)

    assert worker.run_once() is True
    assert store.get_job(owner, job.id).status == JobStatus.CANCELED


def test_startup_requeues_interrupted_job(tmp_path):
    store, job_service, owner, job = queued_job(tmp_path)
    store.claim_next_job()
    worker = JobWorker(store, job_service, lambda: SuccessfulService(), poll_seconds=0.01)

    assert worker.recover() == 1
    assert store.get_job(owner, job.id).status == JobStatus.QUEUED
