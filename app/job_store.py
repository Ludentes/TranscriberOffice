"""Persistent owner and transcription job storage."""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


TERMINAL_STATUSES = (
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELED.value,
)


@dataclass(frozen=True)
class JobRecord:
    id: str
    owner_id: str
    original_filename: str
    audio_path: str | None
    hotwords: str
    status: JobStatus
    progress_text: str
    result_text: str
    result_json: str
    error_message: str | None
    cancel_requested: bool
    created_at: str
    started_at: str | None
    completed_at: str | None
    updated_at: str
    attempt_count: int


@dataclass(frozen=True)
class QueueCounts:
    waiting: int
    running: int

    @property
    def total(self) -> int:
        return self.waiting + self.running


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class JobStore:
    """SQLite repository with owner-scoped public operations."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS owners (
                    id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS transcription_jobs (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES owners(id) ON DELETE CASCADE,
                    original_filename TEXT NOT NULL,
                    audio_path TEXT,
                    hotwords TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed', 'canceled')
                    ),
                    progress_text TEXT NOT NULL DEFAULT '',
                    result_text TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '',
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_owner_created
                    ON transcription_jobs(owner_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON transcription_jobs(status, created_at);
                """
            )

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def resolve_owner(self, token: str) -> str:
        token_hash = self.hash_token(token)
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM owners WHERE token_hash=?", (token_hash,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE owners SET last_seen_at=? WHERE id=?", (now, row["id"])
                )
                return str(row["id"])
            owner_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO owners(id, token_hash, created_at, last_seen_at) VALUES(?,?,?,?)",
                (owner_id, token_hash, now, now),
            )
            return owner_id

    def create_job(
        self,
        owner_id: str,
        original_filename: str,
        audio_path: str,
        hotwords: str,
        *,
        job_id: str | None = None,
    ) -> JobRecord:
        job_id = job_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO transcription_jobs(
                    id, owner_id, original_filename, audio_path, hotwords,
                    status, created_at, updated_at
                ) VALUES(?,?,?,?,?,'queued',?,?)
                """,
                (job_id, owner_id, original_filename, audio_path, hotwords, now, now),
            )
        job = self.get_job(owner_id, job_id)
        assert job is not None
        return job

    def get_job(self, owner_id: str, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM transcription_jobs WHERE id=? AND owner_id=?",
                (job_id, owner_id),
            ).fetchone()
        return self._record(row)

    def get_job_for_worker(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM transcription_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._record(row)

    def list_jobs(self, owner_id: str, limit: int = 100) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM transcription_jobs
                WHERE owner_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (owner_id, limit),
            ).fetchall()
        return [self._record(row) for row in rows if row is not None]

    def claim_next_job(self) -> JobRecord | None:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM transcription_jobs WHERE status='running' LIMIT 1"
            ).fetchone():
                connection.rollback()
                return None
            row = connection.execute(
                """
                SELECT id FROM transcription_jobs WHERE status='queued'
                ORDER BY created_at, rowid LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            cursor = connection.execute(
                """
                UPDATE transcription_jobs
                SET status='running', started_at=?, updated_at=?,
                    attempt_count=attempt_count+1
                WHERE id=? AND status='queued'
                """,
                (now, now, row["id"]),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            claimed = connection.execute(
                "SELECT * FROM transcription_jobs WHERE id=?", (row["id"],)
            ).fetchone()
            connection.commit()
        return self._record(claimed)

    def update_progress(self, job_id: str, progress_text: str) -> bool:
        return self._update_running(
            job_id, "progress_text=?, updated_at=?", (progress_text, _now())
        )

    def complete_job(self, job_id: str, result_text: str, result_json: str) -> bool:
        now = _now()
        return self._update_running(
            job_id,
            """status='completed', result_text=?, result_json=?, progress_text=?,
               completed_at=?, updated_at=?""",
            (result_text, result_json, result_text, now, now),
        )

    def fail_job(self, job_id: str, error_message: str) -> bool:
        now = _now()
        return self._update_running(
            job_id,
            """status='failed', error_message=?, completed_at=?, updated_at=?""",
            (error_message, now, now),
        )

    def cancel_running_job(self, job_id: str) -> bool:
        now = _now()
        return self._update_running(
            job_id,
            """status='canceled', cancel_requested=1, completed_at=?, updated_at=?""",
            (now, now),
        )

    def _update_running(self, job_id: str, assignments: str, values: tuple) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE transcription_jobs SET {assignments} WHERE id=? AND status='running'",
                (*values, job_id),
            )
            return cursor.rowcount == 1

    def request_cancel(self, owner_id: str, job_id: str) -> bool:
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE transcription_jobs
                SET status=CASE WHEN status='queued' THEN 'canceled' ELSE status END,
                    cancel_requested=1,
                    completed_at=CASE WHEN status='queued' THEN ? ELSE completed_at END,
                    updated_at=?
                WHERE id=? AND owner_id=? AND status IN ('queued','running')
                """,
                (now, now, job_id, owner_id),
            )
            return cursor.rowcount == 1

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM transcription_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def requeue_interrupted_jobs(self) -> int:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE transcription_jobs
                SET status='canceled', completed_at=?, updated_at=?
                WHERE status='running' AND cancel_requested=1
                """,
                (now, now),
            )
            cursor = connection.execute(
                """
                UPDATE transcription_jobs
                SET status='queued', started_at=NULL, updated_at=?
                WHERE status='running' AND cancel_requested=0
                """,
                (now,),
            )
            return cursor.rowcount

    def queue_counts(self) -> QueueCounts:
        with self._connect() as connection:
            rows = dict(
                connection.execute(
                    """
                    SELECT status, COUNT(*) FROM transcription_jobs
                    WHERE status IN ('queued','running') GROUP BY status
                    """
                ).fetchall()
            )
        return QueueCounts(waiting=rows.get("queued", 0), running=rows.get("running", 0))

    def list_terminal_with_audio(self) -> list[JobRecord]:
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM transcription_jobs
                WHERE status IN ({placeholders}) AND audio_path IS NOT NULL
                ORDER BY created_at
                """,
                TERMINAL_STATUSES,
            ).fetchall()
        return [self._record(row) for row in rows if row is not None]

    def clear_audio_path(self, job_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE transcription_jobs SET audio_path=NULL, updated_at=? WHERE id=?",
                (_now(), job_id),
            )
            return cursor.rowcount == 1

    def delete_job(self, owner_id: str, job_id: str) -> JobRecord | None:
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM transcription_jobs
                WHERE id=? AND owner_id=? AND status IN ({placeholders})
                """,
                (job_id, owner_id, *TERMINAL_STATUSES),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "DELETE FROM transcription_jobs WHERE id=? AND owner_id=?",
                (job_id, owner_id),
            )
        return self._record(row)

    @staticmethod
    def _record(row: sqlite3.Row | None) -> JobRecord | None:
        if row is None:
            return None
        return JobRecord(
            id=row["id"],
            owner_id=row["owner_id"],
            original_filename=row["original_filename"],
            audio_path=row["audio_path"],
            hotwords=row["hotwords"],
            status=JobStatus(row["status"]),
            progress_text=row["progress_text"],
            result_text=row["result_text"],
            result_json=row["result_json"],
            error_message=row["error_message"],
            cancel_requested=bool(row["cancel_requested"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            updated_at=row["updated_at"],
            attempt_count=row["attempt_count"],
        )
