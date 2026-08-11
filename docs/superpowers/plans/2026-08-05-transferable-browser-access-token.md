# Transferable Browser Access Token Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user deliberately reveal an anonymous browser token and use it in another browser while preserving HttpOnly cookie protection and strict owner isolation.

**Architecture:** Add a non-creating token lookup to the SQLite repository and a focused FastAPI session-transfer router. The router exports the current cookie with no-store headers and imports only known tokens after same-origin validation. The persistent Gradio UI calls these endpoints with same-origin JavaScript and reloads after a successful switch.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, FastAPI/Starlette, Gradio, pytest, FastAPI TestClient.

---

## File Map

- Modify `app/job_store.py`: look up an owner by token without creating one and expose owner counts for regression tests.
- Modify `app/session.py`: centralize protected cookie creation and add the session-transfer router.
- Modify `app/main.py`: mount the session-transfer router.
- Modify `app/ui.py`: add explicit token reveal/import controls to the persistent UI.
- Modify `tests/test_job_store.py`: prove unknown token lookup is non-creating.
- Create `tests/test_session_transfer.py`: test export, import, switching, simultaneous access, and same-origin protection.
- Modify `tests/test_main.py`: verify session-transfer routes are assembled.
- Modify `tests/test_ui.py`: verify security guidance and transfer controls are present.
- Modify `README.md`: document multi-browser access and token handling.

### Task 1: Non-creating owner lookup

**Files:**
- Modify: `app/job_store.py`
- Modify: `tests/test_job_store.py`

- [ ] **Step 1: Write failing lookup tests**

```python
def test_find_owner_by_token_never_creates_owner(tmp_path):
    store = make_store(tmp_path)
    known_token = "a" * 64
    owner_id = store.resolve_owner(known_token)
    before = store.owner_count()

    assert store.find_owner_by_token(known_token) == owner_id
    assert store.find_owner_by_token("b" * 64) is None
    assert store.owner_count() == before
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/python -m pytest tests/test_job_store.py::test_find_owner_by_token_never_creates_owner -v`

Expected: FAIL because `find_owner_by_token` and `owner_count` do not exist.

- [ ] **Step 3: Implement read-only lookup**

Add methods that reuse `hash_token` and perform only `SELECT` statements:

```python
def find_owner_by_token(self, token: str) -> str | None:
    with self._connect() as connection:
        row = connection.execute(
            "SELECT id FROM owners WHERE token_hash=?",
            (self.hash_token(token),),
        ).fetchone()
    return str(row["id"]) if row is not None else None

def owner_count(self) -> int:
    with self._connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM owners").fetchone()[0])
```

- [ ] **Step 4: Run repository tests**

Run: `.venv/bin/python -m pytest tests/test_job_store.py -v`

Expected: all repository tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/job_store.py tests/test_job_store.py
git commit -m "feat: look up existing browser owners safely"
```

### Task 2: Protected session-transfer API

**Files:**
- Modify: `app/session.py`
- Create: `tests/test_session_transfer.py`

- [ ] **Step 1: Write failing API tests**

Build a FastAPI test app with `BrowserSessionMiddleware`, the job router, and
`create_session_router(store, settings)`. Cover export and import:

```python
def test_export_returns_current_token_without_caching(transfer_context):
    store, alice, _ = transfer_context
    alice.get("/api/jobs")
    token = alice.cookies["transcriber_session"]

    response = alice.get("/api/session/token")

    assert response.status_code == 200
    assert response.json() == {"token": token}
    assert response.headers["cache-control"] == "no-store"


def test_known_token_switches_second_browser_and_keeps_first_active(transfer_context):
    store, alice, bob = transfer_context
    alice.get("/api/jobs")
    bob.get("/api/jobs")
    alice_token = alice.cookies["transcriber_session"]
    alice_owner = store.find_owner_by_token(alice_token)
    store.create_job(alice_owner, "shared.mp3", "/private/shared.mp3", "")

    response = bob.post(
        "/api/session/import",
        json={"token": alice_token},
        headers={"Origin": "http://testserver"},
    )

    assert response.status_code == 200
    assert bob.cookies["transcriber_session"] == alice_token
    assert alice.get("/api/jobs").json()[0]["original_filename"] == "shared.mp3"
    assert bob.get("/api/jobs").json()[0]["original_filename"] == "shared.mp3"
```

Also assert malformed/unknown tokens return `400`, do not change the existing
cookie, and do not increase `owner_count`; missing/foreign origins return `403`;
cookie flags follow `SessionConfig`; export rejects a cookie that is not mapped
to `request.state.owner_id`.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_session_transfer.py -v`

Expected: collection FAIL because `create_session_router` does not exist.

- [ ] **Step 3: Centralize cookie construction**

Add `set_session_cookie(response, token, settings)` to `app/session.py` and use
it from both middleware and import:

```python
def set_session_cookie(response: Response, token: str, settings: SessionConfig) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.cookie_max_age_days * 86400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
```

Middleware continues to append only the generated `set-cookie` header.

- [ ] **Step 4: Implement origin validation and router**

Define a Pydantic request model and compare parsed `Origin` scheme/netloc with
`request.url.scheme` and `request.headers["host"]`. Then add:

```python
@router.get("/token")
def export_token(request: Request):
    token = request.cookies.get(settings.cookie_name)
    if not is_valid_session_token(token):
        raise HTTPException(status_code=401, detail="Session token unavailable")
    if store.find_owner_by_token(token) != request.state.owner_id:
        raise HTTPException(status_code=401, detail="Session token unavailable")
    return JSONResponse({"token": token}, headers={"Cache-Control": "no-store"})

@router.post("/import")
def import_token(payload: SessionTokenImport, request: Request):
    require_same_origin(request)
    if not is_valid_session_token(payload.token):
        raise HTTPException(status_code=400, detail="Invalid access token")
    if store.find_owner_by_token(payload.token) is None:
        raise HTTPException(status_code=400, detail="Invalid access token")
    response = JSONResponse({"success": True}, headers={"Cache-Control": "no-store"})
    set_session_cookie(response, payload.token, settings)
    return response
```

- [ ] **Step 5: Run transfer and existing session tests**

Run: `.venv/bin/python -m pytest tests/test_session_transfer.py tests/test_session.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/session.py tests/test_session_transfer.py
git commit -m "feat: export and import browser access tokens"
```

### Task 3: Application and Gradio integration

**Files:**
- Modify: `app/main.py`
- Modify: `app/ui.py`
- Modify: `tests/test_main.py`
- Modify: `tests/test_ui.py`

- [ ] **Step 1: Write failing assembly and UI tests**

In `tests/test_main.py`, assert a request to `/api/session/token` is routed and
returns `200` after the first session-bearing request. In `tests/test_ui.py`,
inspect the persistent Blocks configuration and assert it contains the labels
"Access from another device", "Show token", "Use token", and the warning that
the token grants full history access.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_main.py tests/test_ui.py -v`

Expected: FAIL because the router and controls are not mounted.

- [ ] **Step 3: Mount the router**

Import `create_session_router` in `app/main.py` and include it after middleware
configuration:

```python
app.include_router(create_session_router(store, config.session))
```

- [ ] **Step 4: Add explicit transfer controls**

Inside `_create_persistent_ui`, add a collapsed `gr.Accordion`, warning
Markdown, a read-only export textbox, an import textbox, status HTML, and two
buttons. Use same-origin JavaScript:

```javascript
async () => {
  const response = await fetch('/api/session/token', {
    credentials: 'same-origin', cache: 'no-store'
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || 'Token unavailable');
  return body.token;
}
```

```javascript
async (token) => {
  const response = await fetch('/api/session/import', {
    method: 'POST',
    credentials: 'same-origin',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({token: token.trim()})
  });
  const body = await response.json();
  if (!response.ok) return body.detail || 'Invalid access token';
  window.location.reload();
  return 'Switching history…';
}
```

Do not place either token in a URL, local storage, or persistent Gradio state.

- [ ] **Step 5: Run assembly and UI tests**

Run: `.venv/bin/python -m pytest tests/test_main.py tests/test_ui.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/ui.py tests/test_main.py tests/test_ui.py
git commit -m "feat: add browser token transfer controls"
```

### Task 4: Documentation and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document token transfer**

Explain that "Show token" reveals a password-equivalent secret, "Use token"
switches without merging, previous tokens must be saved before switching, one
token works concurrently in multiple browsers, and HTTPS is required outside
trusted localhost use.

- [ ] **Step 2: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_job_store.py tests/test_session.py tests/test_session_transfer.py tests/test_main.py tests/test_ui.py -v`

Expected: all focused tests PASS.

- [ ] **Step 3: Run complete verification**

Run: `.venv/bin/python -m compileall -q app tests && .venv/bin/python -m pytest tests/ -v && git diff --check`

Expected: compilation and diff checks exit 0 and the full suite has no failures.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain browser token transfer"
```

- [ ] **Step 5: Restart and smoke-test the service**

Stop the existing Uvicorn process, start `.venv/bin/python -m app.main`, then
verify `/`, `/api/health`, `/api/session/token`, and a two-cookie-jar import
flow. Keep the final server process running at `http://localhost:7860`.

