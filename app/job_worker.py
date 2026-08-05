"""Background processing for persistent transcription jobs."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

from app.job_service import JobService
from app.job_store import JobStore
from app.transcribe import TranscriptionResult
from app.transcription_queue import gpu_execution_lock


logger = logging.getLogger(__name__)


def serialize_result(result: TranscriptionResult) -> str:
    return json.dumps(
        {
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "speakers_detected": result.speakers_detected,
            "processing_time_seconds": round(result.processing_time_seconds, 2),
            "segments": result.segments,
            "full_text": result.full_text,
            "error": result.error,
        },
        ensure_ascii=False,
        indent=2,
    )


class JobWorker:
    """Claim and process one durable FIFO job at a time."""

    def __init__(
        self,
        store: JobStore,
        job_service: JobService,
        transcription_service_factory: Callable,
        *,
        poll_seconds: float = 0.5,
    ):
        self.store = store
        self.job_service = job_service
        self.transcription_service_factory = transcription_service_factory
        self.poll_seconds = poll_seconds
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

    def recover(self) -> int:
        requeued = self.store.requeue_interrupted_jobs()
        self.job_service.cleanup_terminal_audio()
        return requeued

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.recover()
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="transcription-job-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def _run(self) -> None:
        while not self._shutdown.is_set():
            if not self.run_once():
                self._shutdown.wait(self.poll_seconds)

    def run_once(self) -> bool:
        job = self.store.claim_next_job()
        if job is None:
            return False

        stop_event = threading.Event()
        terminal = False
        try:
            if not job.audio_path or not Path(job.audio_path).is_file():
                self.store.fail_job(
                    job.id,
                    "Source audio is unavailable. Please upload it again.",
                )
                terminal = True
                return True

            service = self.transcription_service_factory()
            saw_final = False
            with gpu_execution_lock:
                for partial_text, final_result in service.transcribe_stream(
                    audio_path=job.audio_path,
                    hotwords=job.hotwords or None,
                    stop_event=stop_event,
                ):
                    cancel_requested = self.store.is_cancel_requested(job.id)
                    if cancel_requested:
                        stop_event.set()

                    if final_result is None:
                        self.store.update_progress(job.id, partial_text)
                        continue

                    saw_final = True
                    if cancel_requested or final_result.error == "Stopped":
                        self.store.cancel_running_job(job.id)
                    elif final_result.success:
                        self.store.complete_job(
                            job.id,
                            final_result.full_text,
                            serialize_result(final_result),
                        )
                    else:
                        logger.error(
                            "Transcription job %s returned an error: %s",
                            job.id,
                            final_result.error,
                        )
                        self.store.fail_job(
                            job.id,
                            "Transcription failed. Check server logs for details.",
                        )
                    terminal = True
                    break

            if not saw_final:
                if self.store.is_cancel_requested(job.id):
                    self.store.cancel_running_job(job.id)
                else:
                    self.store.fail_job(job.id, "Transcription stopped without a result.")
                terminal = True
        except Exception:
            logger.exception("Persistent transcription job %s failed", job.id)
            self.store.fail_job(
                job.id,
                "Transcription failed. Check server logs for details.",
            )
            terminal = True
        finally:
            if terminal:
                self.job_service.cleanup_audio(job.id)
        return True
