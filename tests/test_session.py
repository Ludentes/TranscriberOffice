from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.config import SessionConfig
from app.job_store import JobStore
from app.session import BrowserSessionMiddleware


def session_client(tmp_path, *, secure=False):
    store = JobStore(tmp_path / "jobs.sqlite3")
    app = FastAPI()
    app.add_middleware(
        BrowserSessionMiddleware,
        store=store,
        settings=SessionConfig(cookie_secure=secure),
    )

    @app.get("/owner")
    def owner(request: Request):
        return {"owner_id": request.state.owner_id}

    return TestClient(app), store


def test_session_cookie_is_created_and_reused(tmp_path):
    client, _ = session_client(tmp_path)

    first = client.get("/owner")
    second = client.get("/owner")

    assert first.json()["owner_id"] == second.json()["owner_id"]
    set_cookie = first.headers["set-cookie"]
    assert "transcriber_session=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Max-Age=31536000" in set_cookie
    assert len(first.headers.get_list("content-length")) == 1
    assert "set-cookie" not in second.headers


def test_invalid_cookie_is_replaced(tmp_path):
    client, _ = session_client(tmp_path)
    client.cookies.set("transcriber_session", "not-a-token")

    response = client.get("/owner")

    assert response.status_code == 200
    assert response.cookies["transcriber_session"] != "not-a-token"
    assert len(response.cookies["transcriber_session"]) == 64


def test_secure_cookie_follows_configuration(tmp_path):
    client, _ = session_client(tmp_path, secure=True)

    response = client.get("/owner")

    assert "Secure" in response.headers["set-cookie"]
