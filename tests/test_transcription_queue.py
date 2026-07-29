import threading
from app.transcription_queue import TranscriptionQueue


def test_queue_runs_jobs_in_fifo_order():
    queue = TranscriptionQueue()
    started = []
    release_first = threading.Event()
    first_active = threading.Event()
    first_ticket = queue.enqueue()
    second_ticket = queue.enqueue()
    third_ticket = queue.enqueue()

    def run_job(job_id, ticket):
        queue.wait(ticket)
        try:
            started.append(job_id)
            if job_id == 1:
                first_active.set()
                release_first.wait(timeout=2)
        finally:
            queue.release(ticket)

    first = threading.Thread(target=run_job, args=(1, first_ticket))
    second = threading.Thread(target=run_job, args=(2, second_ticket))
    third = threading.Thread(target=run_job, args=(3, third_ticket))

    first.start()
    second.start()
    third.start()

    assert first_active.wait(timeout=2)
    assert started == [1]
    release_first.set()

    first.join(timeout=2)
    second.join(timeout=2)
    third.join(timeout=2)

    assert started == [1, 2, 3]


def test_queue_releases_slot_after_error():
    queue = TranscriptionQueue()

    try:
        with queue.slot():
            raise RuntimeError("failed job")
    except RuntimeError:
        pass

    with queue.slot():
        assert True


def test_snapshot_reports_position_and_counts():
    queue = TranscriptionQueue()
    active = queue.enqueue()
    waiting = queue.enqueue()

    assert queue.wait(active)
    snapshot = queue.snapshot(waiting)

    assert snapshot.position == 1
    assert snapshot.waiting_count == 1
    assert snapshot.total_jobs == 2
    assert snapshot.is_active is False

    queue.release(active)
    assert queue.wait(waiting)
    active_snapshot = queue.snapshot(waiting)
    assert active_snapshot.position == 0
    assert active_snapshot.is_active is True
    queue.release(waiting)


def test_wait_timeout_keeps_ticket_in_queue():
    queue = TranscriptionQueue()
    active = queue.enqueue()
    waiting = queue.enqueue()

    assert queue.wait(active)
    assert queue.wait(waiting, timeout=0.01) is False
    assert queue.position(waiting) == 1

    queue.release(active)
    assert queue.wait(waiting)
    queue.release(waiting)
