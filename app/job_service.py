"""Upload and result-file lifecycle for persistent transcription jobs."""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from app.job_store import JobRecord, JobStatus, JobStore, TERMINAL_STATUSES


ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


class JobValidationError(ValueError):
    pass


class JobConflictError(RuntimeError):
    pass


class JobService:
    """Persist validated uploads and clean them after processing."""

    def __init__(self, store: JobStore, data_dir: str | Path, max_file_size_mb: int):
        self.store = store
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def create_job(
        self,
        owner_id: str,
        source_path: str | Path,
        original_filename: str,
        hotwords: str,
    ) -> JobRecord:
        source = Path(source_path)
        extension = Path(original_filename).suffix.lower()
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise JobValidationError(
                f"Unsupported audio format. Expected one of: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )
        if not source.is_file():
            raise JobValidationError("Uploaded audio file is missing")

        job_id = uuid.uuid4().hex
        partial_path = self.upload_dir / f"{job_id}.part"
        final_path = self.upload_dir / f"{job_id}{extension}"
        total = 0
        try:
            with source.open("rb") as input_file, partial_path.open("xb") as output_file:
                while chunk := input_file.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_file_size_bytes:
                        raise JobValidationError(
                            f"Audio file is too large. Maximum size is "
                            f"{self.max_file_size_bytes // (1024 * 1024)}MB"
                        )
                    output_file.write(chunk)
            os.replace(partial_path, final_path)
            return self.store.create_job(
                owner_id,
                Path(original_filename).name,
                str(final_path),
                hotwords.strip(),
                job_id=job_id,
            )
        except Exception:
            partial_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise

    def cleanup_audio(self, job_id: str) -> None:
        job = self.store.get_job_for_worker(job_id)
        if job is None:
            return
        self._unlink_private_path(job.audio_path)
        self.store.clear_audio_path(job_id)

    def cleanup_terminal_audio(self) -> None:
        for job in self.store.list_terminal_with_audio():
            self._unlink_private_path(job.audio_path)
            self.store.clear_audio_path(job.id)

    def delete_job(self, owner_id: str, job_id: str) -> bool:
        job = self.store.get_job(owner_id, job_id)
        if job is None:
            return False
        if job.status.value not in TERMINAL_STATUSES:
            raise JobConflictError("Only completed, failed, or canceled jobs can be deleted")
        deleted = self.store.delete_job(owner_id, job_id)
        if deleted is None:
            return False
        self._unlink_private_path(deleted.audio_path)
        return True

    def _unlink_private_path(self, audio_path: str | None) -> None:
        if not audio_path:
            return
        path = Path(audio_path)
        try:
            path.resolve().relative_to(self.upload_dir.resolve())
        except ValueError:
            return
        path.unlink(missing_ok=True)

    @staticmethod
    def download_name(original_filename: str, extension: str) -> str:
        stem = Path(original_filename).stem
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
        safe_stem = safe_stem[:80] or "transcription"
        safe_extension = "json" if extension == "json" else "txt"
        return f"{safe_stem}_transcript.{safe_extension}"
