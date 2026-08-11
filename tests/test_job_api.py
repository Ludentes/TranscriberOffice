import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import SessionConfig
from app.job_api import create_job_router
from app.job_service import JobService
from app.job_store import JobStore
from app.session import BrowserSessionMiddleware


@pytest.fixture
def api_context(tmp_path):
    store = JobStore(tmp_path / "data" / "jobs.sqlite3")
    service = JobService(store, tmp_path / "data", max_file_size_mb=1)
    app = FastAPI()
    app.add_middleware(
        BrowserSessionMiddleware,
        store=store,
        settings=SessionConfig(),
    )
    app.include_router(create_job_router(store, service))
    alice = TestClient(app)
    bob = TestClient(app)
    alice.get("/api/jobs")
    bob.get("/api/jobs")
    return store, service, alice, bob


def owner_for(client, store):
    return store.resolve_owner(client.cookies["transcriber_session"])


def test_create_and_list_are_scoped_to_browser(api_context):
    store, _, alice, bob = api_context

    created = alice.post(
        "/api/jobs",
        files={"file": ("meeting.mp3", b"audio", "audio/mpeg")},
        data={"hotwords": "Project X"},
    )

    assert created.status_code == 202
    job = created.json()
    assert job["original_filename"] == "meeting.mp3"
    assert job["status"] == "queued"
    assert "owner_id" not in job
    assert "audio_path" not in job
    assert [item["id"] for item in alice.get("/api/jobs").json()] == [job["id"]]
    assert bob.get("/api/jobs").json() == []


def test_other_browser_cannot_read_cancel_delete_or_download(api_context):
    store, _, alice, bob = api_context
    owner = owner_for(alice, store)
    job = store.create_job(owner, "secret.mp3", "/private/secret.mp3", "")

    endpoints = [
        ("get", f"/api/jobs/{job.id}"),
        ("post", f"/api/jobs/{job.id}/cancel"),
        ("delete", f"/api/jobs/{job.id}"),
        ("get", f"/api/jobs/{job.id}/download.txt"),
        ("get", f"/api/jobs/{job.id}/download.json"),
    ]

    for method, path in endpoints:
        assert getattr(bob, method)(path).status_code == 404


def test_cancel_and_delete_owned_job(api_context):
    store, _, alice, _ = api_context
    owner = owner_for(alice, store)
    job = store.create_job(owner, "cancel.mp3", "/private/cancel.mp3", "")

    canceled = alice.post(f"/api/jobs/{job.id}/cancel")

    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"
    assert alice.delete(f"/api/jobs/{job.id}").status_code == 204
    assert alice.get(f"/api/jobs/{job.id}").status_code == 404


def test_completed_job_downloads_text_and_json(api_context):
    store, _, alice, _ = api_context
    owner = owner_for(alice, store)
    job = store.create_job(owner, "Quarter report?.mp3", "/private/report.mp3", "")
    store.claim_next_job()
    payload = '{"success": true, "full_text": "hello"}'
    store.complete_job(job.id, "hello", payload)

    text = alice.get(f"/api/jobs/{job.id}/download.txt")
    structured = alice.get(f"/api/jobs/{job.id}/download.json")

    assert text.status_code == 200
    assert text.text == "hello"
    assert text.headers["content-disposition"] == 'attachment; filename="Quarter_report_transcript.txt"'
    assert structured.status_code == 200
    assert structured.json() == json.loads(payload)


def test_active_job_cannot_be_deleted_or_downloaded(api_context):
    store, _, alice, _ = api_context
    owner = owner_for(alice, store)
    job = store.create_job(owner, "active.mp3", "/private/active.mp3", "")

    assert alice.delete(f"/api/jobs/{job.id}").status_code == 409
    assert alice.get(f"/api/jobs/{job.id}/download.txt").status_code == 409


def test_queue_endpoint_returns_counts_only(api_context):
    store, _, alice, _ = api_context
    owner = owner_for(alice, store)
    store.create_job(owner, "meeting.mp3", "/private/meeting.mp3", "")

    assert alice.get("/api/jobs/queue").json() == {
        "waiting_count": 1,
        "running_count": 0,
        "total_jobs": 1,
    }
