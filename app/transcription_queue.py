"""A process-wide FIFO queue for GPU transcription jobs."""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Generator


# Persistent jobs and the legacy synchronous API share the same GPU/model.
gpu_execution_lock = threading.Lock()


@dataclass(frozen=True)
class QueueTicket:
    """Opaque ticket identifying a job's place in the queue."""

    number: int


@dataclass(frozen=True)
class QueueSnapshot:
    """Current queue information shown to a waiting user."""

    position: int
    waiting_count: int
    total_jobs: int
    is_active: bool


@dataclass(frozen=True)
class GlobalQueueSnapshot:
    """Queue information available to visitors without their own ticket."""

    waiting_count: int
    total_jobs: int


class TranscriptionQueue:
    """Serialize transcription jobs in the order they were submitted."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting: deque[QueueTicket] = deque()
        self._active: QueueTicket | None = None
        self._next_number = 1

    def enqueue(self) -> QueueTicket:
        """Add a job to the end of the queue without waiting for its turn."""
        with self._condition:
            ticket = QueueTicket(self._next_number)
            self._next_number += 1
            self._waiting.append(ticket)
            self._condition.notify_all()
            return ticket

    def wait(self, ticket: QueueTicket, timeout: float | None = None) -> bool:
        """Wait for a turn, returning ``False`` when *timeout* expires."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if ticket not in self._waiting:
                    return False
                if self._active is None and self._waiting[0] == ticket:
                    self._waiting.popleft()
                    self._active = ticket
                    return True

                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)

    def release(self, ticket: QueueTicket) -> None:
        """Release an active ticket, allowing the next job to start."""
        with self._condition:
            if self._active != ticket:
                raise RuntimeError("Cannot release a transcription ticket that is not active")
            self._active = None
            self._condition.notify_all()

    def cancel(self, ticket: QueueTicket) -> None:
        """Remove a ticket which has not started yet."""
        with self._condition:
            try:
                self._waiting.remove(ticket)
            except ValueError:
                return
            self._condition.notify_all()

    @contextmanager
    def slot(self) -> Generator[QueueTicket, None, None]:
        """Enqueue a job and hold the single transcription slot."""
        ticket = self.enqueue()
        acquired = False
        try:
            acquired = self.wait(ticket)
            if not acquired:
                raise RuntimeError("Transcription ticket was canceled before acquisition")
            yield ticket
        finally:
            if acquired:
                self.release(ticket)
            else:
                self.cancel(ticket)

    def position(self, ticket: QueueTicket) -> int:
        """Return a one-based queue position, or zero for the active job."""
        with self._condition:
            if self._active == ticket:
                return 0
            try:
                return list(self._waiting).index(ticket) + 1
            except ValueError:
                return -1

    def snapshot(self, ticket: QueueTicket) -> QueueSnapshot:
        """Return the current position and queue counts for one ticket."""
        with self._condition:
            waiting_count = len(self._waiting)
            total_jobs = waiting_count + (1 if self._active is not None else 0)

            if self._active == ticket:
                return QueueSnapshot(
                    position=0,
                    waiting_count=waiting_count,
                    total_jobs=total_jobs,
                    is_active=True,
                )

            try:
                index = list(self._waiting).index(ticket)
            except ValueError:
                return QueueSnapshot(
                    position=-1,
                    waiting_count=waiting_count,
                    total_jobs=total_jobs,
                    is_active=False,
                )

            return QueueSnapshot(
                position=index + 1,
                waiting_count=waiting_count,
                total_jobs=total_jobs,
                is_active=False,
            )

    def global_snapshot(self) -> GlobalQueueSnapshot:
        """Return a live queue summary for page load and manual refresh."""
        with self._condition:
            waiting_count = len(self._waiting)
            return GlobalQueueSnapshot(
                waiting_count=waiting_count,
                total_jobs=waiting_count + (1 if self._active is not None else 0),
            )


transcription_queue = TranscriptionQueue()
