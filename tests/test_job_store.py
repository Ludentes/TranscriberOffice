import hashlib
import sqlite3

from app.job_store import JobStatus, JobStore


def make_store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite3")


def test_owner_token_is_hashed(tmp_path):
    store = make_store(tmp_path)

    owner_id = store.resolve_owner("a" * 64)

    assert store.resolve_owner("a" * 64) == owner_id
    with sqlite3.connect(store.db_path) as connection:
        row = connection.execute("SELECT token_hash FROM owners").fetchone()
    assert row[0] != "a" * 64
    assert row[0] == hashlib.sha256(("a" * 64).encode()).hexdigest()


def test_find_owner_by_token_never_creates_owner(tmp_path):
    store = make_store(tmp_path)
    known_token = "a" * 64
    owner_id = store.resolve_owner(known_token)
    before = store.owner_count()

    assert store.find_owner_by_token(known_token) == owner_id
    assert store.find_owner_by_token("b" * 64) is None
    assert store.owner_count() == before


def test_claims_jobs_fifo_and_only_one_runs(tmp_path):
    store = make_store(tmp_path)
    owner_id = store.resolve_owner("b" * 64)
    first = store.create_job(owner_id, "first.mp3", "/private/first.mp3", "")
    second = store.create_job(owner_id, "second.mp3", "/private/second.mp3", "")

    assert store.claim_next_job().id == first.id
    assert store.claim_next_job() is None

    assert store.complete_job(first.id, "text", '{"success": true}') is True
    assert store.claim_next_job().id == second.id


def test_foreign_owner_cannot_read_or_mutate_job(tmp_path):
    store = make_store(tmp_path)
    alice = store.resolve_owner("c" * 64)
    bob = store.resolve_owner("d" * 64)
    job = store.create_job(alice, "meeting.mp3", "/private/meeting.mp3", "")

    assert store.get_job(bob, job.id) is None
    assert store.request_cancel(bob, job.id) is False
    assert store.delete_job(bob, job.id) is None
    assert store.list_jobs(bob) == []


def test_list_jobs_is_newest_first_and_owner_scoped(tmp_path):
    store = make_store(tmp_path)
    alice = store.resolve_owner("e" * 64)
    bob = store.resolve_owner("f" * 64)
    first = store.create_job(alice, "first.mp3", "/private/first.mp3", "")
    second = store.create_job(alice, "second.mp3", "/private/second.mp3", "")
    store.create_job(bob, "secret.mp3", "/private/secret.mp3", "")

    assert [job.id for job in store.list_jobs(alice)] == [second.id, first.id]


def test_cancel_and_terminal_transitions_are_conditional(tmp_path):
    store = make_store(tmp_path)
    owner = store.resolve_owner("1" * 64)
    queued = store.create_job(owner, "queued.mp3", "/private/queued.mp3", "")

    assert store.request_cancel(owner, queued.id) is True
    assert store.get_job(owner, queued.id).status == JobStatus.CANCELED
    assert store.complete_job(queued.id, "late", "{}") is False

    running = store.create_job(owner, "running.mp3", "/private/running.mp3", "")
    assert store.claim_next_job().id == running.id
    assert store.request_cancel(owner, running.id) is True
    assert store.is_cancel_requested(running.id) is True
    assert store.cancel_running_job(running.id) is True
    assert store.get_job(owner, running.id).status == JobStatus.CANCELED


def test_startup_requeues_running_jobs_and_counts_queue(tmp_path):
    store = make_store(tmp_path)
    owner = store.resolve_owner("2" * 64)
    first = store.create_job(owner, "first.mp3", "/private/first.mp3", "")
    store.create_job(owner, "second.mp3", "/private/second.mp3", "")
    assert store.claim_next_job().id == first.id

    assert store.requeue_interrupted_jobs() == 1
    counts = store.queue_counts()

    assert counts.running == 0
    assert counts.waiting == 2
    assert counts.total == 2
    assert store.get_job(owner, first.id).attempt_count == 1


def test_progress_failure_cleanup_and_terminal_delete(tmp_path):
    store = make_store(tmp_path)
    owner = store.resolve_owner("3" * 64)
    job = store.create_job(owner, "meeting.mp3", "/private/meeting.mp3", "words")
    store.claim_next_job()

    assert store.update_progress(job.id, "halfway") is True
    assert store.get_job(owner, job.id).progress_text == "halfway"
    assert store.fail_job(job.id, "safe error") is True
    failed = store.get_job(owner, job.id)
    assert failed.error_message == "safe error"
    assert store.list_terminal_with_audio() == [failed]
    assert store.clear_audio_path(job.id) is True
    assert store.get_job(owner, job.id).audio_path is None

    deleted = store.delete_job(owner, job.id)
    assert deleted.id == job.id
    assert store.get_job(owner, job.id) is None
