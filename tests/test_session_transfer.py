import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import SessionConfig
from app.job_api import create_job_router
from app.job_service import JobService
from app.job_store import JobStore
from app.session import BrowserSessionMiddleware, create_session_router


def make_transfer_app(tmp_path, settings=None):
    settings = settings or SessionConfig()
    store = JobStore(tmp_path / "data" / "jobs.sqlite3")
    service = JobService(store, tmp_path / "data", max_file_size_mb=1)
    app = FastAPI()
    app.add_middleware(BrowserSessionMiddleware, store=store, settings=settings)
    app.include_router(create_job_router(store, service))
    app.include_router(create_session_router(store, settings))
    return app, store


@pytest.fixture
def transfer_context(tmp_path):
    app, store = make_transfer_app(tmp_path)
    alice = TestClient(app)
    bob = TestClient(app)
    alice.get("/api/jobs")
    bob.get("/api/jobs")
    return store, alice, bob


def test_export_returns_current_token_without_caching(transfer_context):
    _, alice, _ = transfer_context
    token = alice.cookies["transcriber_session"]

    response = alice.get("/api/session/token")

    assert response.status_code == 200
    assert response.json() == {"token": token}
    assert response.headers["cache-control"] == "no-store"


def test_known_token_switches_second_browser_and_keeps_first_active(transfer_context):
    store, alice, bob = transfer_context
    alice_token = alice.cookies["transcriber_session"]
    alice_owner = store.find_owner_by_token(alice_token)
    store.create_job(alice_owner, "shared.mp3", "/private/shared.mp3", "")

    response = bob.post(
        "/api/session/import",
        json={"token": alice_token},
        headers={"Origin": "http://testserver"},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert bob.cookies["transcriber_session"] == alice_token
    assert alice.get("/api/jobs").json()[0]["original_filename"] == "shared.mp3"
    assert bob.get("/api/jobs").json()[0]["original_filename"] == "shared.mp3"


def test_import_switches_without_merging_histories(transfer_context):
    store, alice, bob = transfer_context
    alice_token = alice.cookies["transcriber_session"]
    bob_token = bob.cookies["transcriber_session"]
    alice_owner = store.find_owner_by_token(alice_token)
    bob_owner = store.find_owner_by_token(bob_token)
    store.create_job(alice_owner, "alice.mp3", "/private/alice.mp3", "")
    store.create_job(bob_owner, "bob.mp3", "/private/bob.mp3", "")

    bob.post(
        "/api/session/import",
        json={"token": alice_token},
        headers={"Origin": "http://testserver"},
    )

    assert [job["original_filename"] for job in bob.get("/api/jobs").json()] == ["alice.mp3"]
    old_bob = TestClient(bob.app)
    old_bob.cookies.set("transcriber_session", bob_token)
    assert [job["original_filename"] for job in old_bob.get("/api/jobs").json()] == ["bob.mp3"]


@pytest.mark.parametrize("token", ["short", "G" * 64, "f" * 64])
def test_invalid_or_unknown_token_does_not_change_cookie_or_create_owner(
    transfer_context, token
):
    store, _, bob = transfer_context
    current_token = bob.cookies["transcriber_session"]
    before = store.owner_count()

    response = bob.post(
        "/api/session/import",
        json={"token": token},
        headers={"Origin": "http://testserver"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid access token"}
    assert bob.cookies["transcriber_session"] == current_token
    assert store.owner_count() == before


@pytest.mark.parametrize("headers", [{}, {"Origin": "https://evil.example"}])
def test_import_rejects_missing_or_foreign_origin(transfer_context, headers):
    _, alice, bob = transfer_context
    current_token = bob.cookies["transcriber_session"]

    response = bob.post(
        "/api/session/import",
        json={"token": alice.cookies["transcriber_session"]},
        headers=headers,
    )

    assert response.status_code == 403
    assert bob.cookies["transcriber_session"] == current_token


def test_import_cookie_uses_protected_flags(tmp_path):
    settings = SessionConfig(cookie_secure=True, cookie_max_age_days=30)
    app, _ = make_transfer_app(tmp_path, settings)
    alice = TestClient(app, base_url="https://testserver")
    bob = TestClient(app, base_url="https://testserver")
    alice.get("/api/jobs")
    bob.get("/api/jobs")

    response = bob.post(
        "/api/session/import",
        json={"token": alice.cookies["transcriber_session"]},
        headers={"Origin": "https://testserver"},
    )

    set_cookie = response.headers["set-cookie"]
    assert response.status_code == 200
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Max-Age=2592000" in set_cookie


def test_import_on_first_request_keeps_imported_cookie(tmp_path):
    app, _ = make_transfer_app(tmp_path)
    alice = TestClient(app)
    alice.get("/api/jobs")
    token = alice.cookies["transcriber_session"]
    fresh_browser = TestClient(app)

    response = fresh_browser.post(
        "/api/session/import",
        json={"token": token},
        headers={"Origin": "http://testserver"},
    )

    assert response.status_code == 200
    assert fresh_browser.cookies["transcriber_session"] == token
