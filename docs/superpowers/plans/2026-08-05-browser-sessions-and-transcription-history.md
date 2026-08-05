# Browser Sessions and Transcription History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist anonymous browser sessions, background transcription jobs, and owner-isolated downloadable history across page and server restarts.

**Architecture:** FastAPI middleware maps an opaque HttpOnly cookie to an owner stored in SQLite. A focused repository owns persistent job state, while one background worker claims queued jobs and writes progress/results independently of Gradio requests. Gradio callbacks use the owner placed on each request and render only owner-scoped history.

**Tech Stack:** Python 3.10+, FastAPI/Starlette, Gradio, stdlib `sqlite3`, `threading`, `hashlib`, `secrets`, pytest, FastAPI TestClient.

---

## File Map

- Create `app/job_store.py`: SQLite schema, owner records, job records, state transitions, and owner-scoped queries.
- Create `app/session.py`: cookie constants, token validation/hashing, and FastAPI session middleware.
- Create `app/job_service.py`: validated upload persistence, job creation, deletion, and result download helpers.
- Create `app/job_worker.py`: persistent FIFO worker, restart recovery, cancellation, progress persistence, and audio cleanup.
- Create `app/job_api.py`: owner-scoped asynchronous job/history endpoints and downloads.
- Modify `app/config.py`: storage and cookie configuration.
- Modify `app/main.py`: initialize shared components, lifespan worker, middleware, routers, and UI dependencies.
- Modify `app/api.py`: share the GPU execution lock with the background worker.
- Modify `app/transcription_queue.py`: expose the single process-wide GPU lock without relying on request lifetime.
- Modify `app/ui.py`: submit persistent jobs and restore current/history state from the request owner.
- Modify `config.yaml`, `.gitignore`, `compose.yaml`: data directory settings and persistence.
- Create `tests/test_job_store.py`, `tests/test_session.py`, `tests/test_job_service.py`, `tests/test_job_worker.py`, `tests/test_job_api.py`.
- Modify `tests/test_config.py`, `tests/test_api.py`, `tests/test_ui.py`, `README.md`.

### Task 1: Storage and session configuration

**Files:**
- Modify: `app/config.py`
- Modify: `config.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Add assertions for defaults and YAML overrides:

```python
def test_storage_and_session_defaults(tmp_path):
    from app.config import load_config

    config = load_config(tmp_path / "missing.yaml")
    assert config.storage.data_dir == "./data"
    assert config.session.cookie_name == "transcriber_session"
    assert config.session.cookie_secure is False
    assert config.session.cookie_max_age_days == 365


def test_load_storage_and_session_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "storage:\n  data_dir: /srv/transcriber\n"
        "session:\n  cookie_secure: true\n  cookie_max_age_days: 30\n"
    )
    from app.config import load_config

    config = load_config(config_file)
    assert config.storage.data_dir == "/srv/transcriber"
    assert config.session.cookie_secure is True
    assert config.session.cookie_max_age_days == 30
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest tests/test_config.py -v`

Expected: FAIL because `AppConfig` has no `storage` or `session` attributes.

- [ ] **Step 3: Add focused configuration dataclasses**

Add to `app/config.py`:

```python
@dataclass
class StorageConfig:
    data_dir: str = "./data"


@dataclass
class SessionConfig:
    cookie_name: str = "transcriber_session"
    cookie_secure: bool = False
    cookie_max_age_days: int = 365


@dataclass
class AppConfig:
    server: ServerConfig
    model: ModelConfig
    transcription: TranscriptionConfig
    storage: StorageConfig
    session: SessionConfig
```

Construct both dataclasses in the missing-file and YAML branches of `load_config`.
Add matching `storage` and `session` sections to `config.yaml`.

- [ ] **Step 4: Run configuration tests**

Run: `pytest tests/test_config.py -v`

Expected: all configuration tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py config.yaml tests/test_config.py
git commit -m "feat: configure persistent session storage"
```

### Task 2: SQLite owner and job repository

**Files:**
- Create: `app/job_store.py`
- Create: `tests/test_job_store.py`

- [ ] **Step 1: Write failing owner and state-transition tests**

Create tests using `JobStore(tmp_path / "jobs.sqlite3")` that verify:

```python
def test_owner_token_is_hashed(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    owner_id = store.resolve_owner("a" * 64)
    assert store.resolve_owner("a" * 64) == owner_id
    with sqlite3.connect(store.db_path) as connection:
        row = connection.execute("SELECT token_hash FROM owners").fetchone()
    assert row[0] != "a" * 64
    assert row[0] == hashlib.sha256(("a" * 64).encode()).hexdigest()


def test_claims_jobs_fifo_and_only_once(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    owner_id = store.resolve_owner("b" * 64)
    first = store.create_job(owner_id, "first.mp3", "/private/first.mp3", "")
    second = store.create_job(owner_id, "second.mp3", "/private/second.mp3", "")
    assert store.claim_next_job().id == first.id
    assert store.claim_next_job() is None
    store.complete_job(first.id, "text", '{"success": true}')
    assert store.claim_next_job().id == second.id


def test_foreign_owner_cannot_read_or_mutate_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    alice = store.resolve_owner("c" * 64)
    bob = store.resolve_owner("d" * 64)
    job = store.create_job(alice, "meeting.mp3", "/private/meeting.mp3", "")
    assert store.get_job(bob, job.id) is None
    assert store.request_cancel(bob, job.id) is False
    assert store.delete_job(bob, job.id) is None
```

Also cover list ordering, startup requeue, queued cancellation, running
cancellation flags, terminal transition protection, and queue counts.

- [ ] **Step 2: Run the repository tests and verify failure**

Run: `pytest tests/test_job_store.py -v`

Expected: collection FAIL because `app.job_store` does not exist.

- [ ] **Step 3: Implement schema and record types**

Create `JobStatus`, `JobRecord`, and `JobStore`. Initialize the schema in the
constructor. Each connection must execute:

```python
connection.execute("PRAGMA journal_mode=WAL")
connection.execute("PRAGMA foreign_keys=ON")
connection.execute("PRAGMA busy_timeout=5000")
```

Use an atomic conditional claim:

```python
connection.execute("BEGIN IMMEDIATE")
row = connection.execute(
    "SELECT id FROM transcription_jobs WHERE status = 'queued' "
    "ORDER BY created_at, id LIMIT 1"
).fetchone()
if row is None or connection.execute(
    "SELECT 1 FROM transcription_jobs WHERE status = 'running' LIMIT 1"
).fetchone():
    connection.rollback()
    return None
connection.execute(
    "UPDATE transcription_jobs SET status='running', started_at=?, "
    "updated_at=?, attempt_count=attempt_count+1 WHERE id=? AND status='queued'",
    (now, now, row[0]),
)
connection.commit()
```

All user-facing methods accept `owner_id`; worker methods accept only `job_id`
and enforce allowed source states in SQL `WHERE` clauses.

- [ ] **Step 4: Run repository tests**

Run: `pytest tests/test_job_store.py -v`

Expected: all repository tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/job_store.py tests/test_job_store.py
git commit -m "feat: persist owners and transcription jobs"
```

### Task 3: Protected browser-session middleware

**Files:**
- Create: `app/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write failing middleware tests**

Build a minimal FastAPI app with `BrowserSessionMiddleware` and a route returning
`request.state.owner_id`. Verify the first request sets a cookie containing
`HttpOnly` and `SameSite=lax`, the second request keeps the same owner, an
invalid cookie is replaced, and `Secure` follows configuration.

```python
def test_session_cookie_is_reused(tmp_path):
    client, store = session_client(tmp_path)
    first = client.get("/owner")
    second = client.get("/owner")
    assert first.json()["owner_id"] == second.json()["owner_id"]
    assert "HttpOnly" in first.headers["set-cookie"]
    assert "SameSite=lax" in first.headers["set-cookie"]
```

- [ ] **Step 2: Run the middleware tests and verify failure**

Run: `pytest tests/test_session.py -v`

Expected: collection FAIL because `app.session` does not exist.

- [ ] **Step 3: Implement token creation and middleware**

Use `secrets.token_hex(32)` and require exactly 64 lowercase hexadecimal
characters. Middleware resolves the owner through `JobStore`, writes
`request.state.owner_id`, and sets a replacement cookie only when it generated
a token:

```python
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

- [ ] **Step 4: Run middleware and repository tests**

Run: `pytest tests/test_session.py tests/test_job_store.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/session.py tests/test_session.py
git commit -m "feat: identify browsers with protected cookies"
```

### Task 4: Upload lifecycle and owner-scoped job service

**Files:**
- Create: `app/job_service.py`
- Create: `tests/test_job_service.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing service tests**

Test allowed extensions, maximum byte size, generated paths, rollback cleanup,
terminal-only deletion, and safe filenames. The success case must assert the
saved path is inside `<data_dir>/uploads` and contains the job UUID rather than
the original basename.

```python
def test_create_job_copies_audio_to_private_generated_path(tmp_path):
    service, store = make_service(tmp_path)
    source = tmp_path / "My Meeting.mp3"
    source.write_bytes(b"audio")
    job = service.create_job("owner", source, source.name, "Project X")
    saved = Path(job.audio_path)
    assert saved.parent == tmp_path / "data" / "uploads"
    assert saved.name == f"{job.id}.mp3"
    assert saved.read_bytes() == b"audio"
```

- [ ] **Step 2: Run the service tests and verify failure**

Run: `pytest tests/test_job_service.py -v`

Expected: collection FAIL because `app.job_service` does not exist.

- [ ] **Step 3: Implement `JobService`**

Implement `create_job`, `delete_job`, `download_name`, and `cleanup_audio`.
Validate `{.mp3,.wav,.m4a,.ogg,.flac}`, stream-copy in 1 MiB blocks while
enforcing `max_file_size_mb`, write to `<job-id>.part`, and atomically rename to
the final generated path. On every exception, unlink both partial and final
paths before re-raising a user-safe `JobValidationError` where appropriate.

- [ ] **Step 4: Ignore and mount persistent data**

Add `data/` to `.gitignore`. Do not add a database file or uploaded fixture to
Git.

- [ ] **Step 5: Run service tests**

Run: `pytest tests/test_job_service.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/job_service.py tests/test_job_service.py .gitignore
git commit -m "feat: store uploads for persistent jobs"
```

### Task 5: Persistent background worker and shared GPU lock

**Files:**
- Create: `app/job_worker.py`
- Create: `tests/test_job_worker.py`
- Modify: `app/transcription_queue.py`
- Modify: `app/api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing worker tests**

Use a fake transcription service yielding one partial update and a successful
`TranscriptionResult`. Verify progress/result persistence, cleanup, cancellation,
failure continuation, and startup requeue.

```python
def test_worker_persists_result_and_removes_audio(tmp_path):
    store, job_service, job = queued_job(tmp_path)
    worker = JobWorker(store, job_service, lambda: successful_service(), poll_seconds=0.01)
    assert worker.run_once() is True
    saved = store.get_job(job.owner_id, job.id)
    assert saved.status == JobStatus.COMPLETED
    assert saved.result_text == "final transcript"
    assert not Path(job.audio_path).exists()
```

- [ ] **Step 2: Run worker tests and verify failure**

Run: `pytest tests/test_job_worker.py -v`

Expected: collection FAIL because `app.job_worker` does not exist.

- [ ] **Step 3: Expose one shared GPU execution lock**

Add `gpu_execution_lock = threading.Lock()` to `app/transcription_queue.py`.
Wrap the existing synchronous `/api/transcribe` model call with this lock while
keeping its current response contract and cleanup behavior.

- [ ] **Step 4: Implement `JobWorker`**

Provide `start`, `stop`, `_run`, and deterministic `run_once`. `start` first
calls `store.requeue_interrupted_jobs()`. `run_once` claims one job, checks the
audio path, creates a per-job `threading.Event`, acquires
`gpu_execution_lock`, then consumes `transcribe_stream`:

```python
for partial_text, final_result in service.transcribe_stream(
    audio_path=job.audio_path,
    hotwords=job.hotwords or None,
    stop_event=cancel_event,
):
    if store.is_cancel_requested(job.id):
        cancel_event.set()
    if final_result is None:
        store.update_progress(job.id, partial_text)
    elif final_result.success:
        store.complete_job(job.id, final_result.full_text, serialize_result(final_result))
    elif final_result.error == "Stopped":
        store.cancel_running_job(job.id)
    else:
        store.fail_job(job.id, sanitize_error(final_result.error))
```

Catch exceptions, store a generic failure message, log the traceback, and
always call `job_service.cleanup_audio(job.id)` after a terminal transition.
During `start`, also scan terminal jobs with a non-null `audio_path`, unlink the
file best-effort, and clear the stored path so a previous cleanup failure is
repaired after restart.

- [ ] **Step 5: Run worker and synchronous API tests**

Run: `pytest tests/test_job_worker.py tests/test_api.py -v`

Expected: all tests PASS and the synchronous response schema is unchanged.

- [ ] **Step 6: Commit**

```bash
git add app/job_worker.py tests/test_job_worker.py app/transcription_queue.py app/api.py tests/test_api.py
git commit -m "feat: process persistent jobs in background"
```

### Task 6: Owner-isolated history API and downloads

**Files:**
- Create: `app/job_api.py`
- Create: `tests/test_job_api.py`

- [ ] **Step 1: Write failing API isolation tests**

Create an app fixture with the session middleware and dependency overrides for
temporary `JobStore`/`JobService`. Use two TestClient instances with different
cookie jars. Verify create, list, detail, cancel, delete, TXT download, and JSON
download. For every detail/mutation/download endpoint, assert the second client
receives `404` for the first client's job.

```python
def test_other_browser_cannot_download_job(app_clients, completed_job):
    alice, bob = app_clients
    assert alice.get(f"/api/jobs/{completed_job.id}/download.txt").status_code == 200
    assert bob.get(f"/api/jobs/{completed_job.id}/download.txt").status_code == 404
```

- [ ] **Step 2: Run API tests and verify failure**

Run: `pytest tests/test_job_api.py -v`

Expected: collection FAIL because `app.job_api` does not exist.

- [ ] **Step 3: Implement request dependencies and schemas**

Define `JobSummary`, `JobDetail`, and `QueueStatus` Pydantic models. Resolve
`owner_id` exclusively from `request.state.owner_id`; never accept it as a path,
query, form, or JSON field.

- [ ] **Step 4: Implement endpoints**

Add:

```text
POST   /api/jobs
GET    /api/jobs
GET    /api/jobs/{job_id}
POST   /api/jobs/{job_id}/cancel
DELETE /api/jobs/{job_id}
GET    /api/jobs/{job_id}/download.txt
GET    /api/jobs/{job_id}/download.json
GET    /api/jobs/queue
```

Return `409` when a requested transition conflicts with the current owned job
state, `400` for invalid uploads, and `404` for absent or foreign jobs. Download
routes require `completed`, use stored bytes, and set sanitized attachment names.

- [ ] **Step 5: Run job API tests**

Run: `pytest tests/test_job_api.py -v`

Expected: all tests PASS, including cross-owner access checks.

- [ ] **Step 6: Commit**

```bash
git add app/job_api.py tests/test_job_api.py
git commit -m "feat: expose private transcription history API"
```

### Task 7: Application lifecycle and persistent Docker volume

**Files:**
- Modify: `app/main.py`
- Modify: `compose.yaml`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write failing lifecycle tests**

Patch the worker and assert FastAPI lifespan calls `start()` once and `stop()`
once. Assert a request through the assembled app receives a session cookie and
that the new job router is mounted.

- [ ] **Step 2: Run lifecycle tests and verify failure**

Run: `pytest tests/test_main.py -v`

Expected: FAIL because `create_app` does not configure persistent components.

- [ ] **Step 3: Add an application component factory and lifespan**

In `app/main.py`, create the database/upload directories, instantiate
`JobStore`, `JobService`, and `JobWorker`, store them in `app.state`, install
`BrowserSessionMiddleware`, include both routers, and run the worker through an
`asynccontextmanager` lifespan. Pass the same store/service into `create_ui`.

- [ ] **Step 4: Persist data in Compose**

Add `./data:/app/data` under `services.transcriber.volumes`. Keep the existing
model and read-only source mounts.

- [ ] **Step 5: Run lifecycle and API tests**

Run: `pytest tests/test_main.py tests/test_session.py tests/test_job_api.py -v`

Expected: all tests PASS with no worker thread leaked after TestClient closes.

- [ ] **Step 6: Commit**

```bash
git add app/main.py compose.yaml tests/test_main.py
git commit -m "feat: wire persistent jobs into app lifecycle"
```

### Task 8: Gradio current task and history UI

**Files:**
- Modify: `app/ui.py`
- Modify: `tests/test_ui.py`

- [ ] **Step 1: Write failing UI helper tests**

Extract testable helpers and verify owner resolution, dropdown choices, selected
job rendering, queue labels, and protected download links:

```python
def test_history_choices_do_not_include_foreign_jobs(store_with_two_owners):
    store, alice, bob = store_with_two_owners
    choices = build_history_choices(store.list_jobs(alice))
    assert any("alice.mp3" in label for label, _ in choices)
    assert all("bob.mp3" not in label for label, _ in choices)


def test_download_links_use_owned_job_endpoint(completed_job):
    markdown = build_download_links(completed_job)
    assert f"/api/jobs/{completed_job.id}/download.txt" in markdown
    assert f"/api/jobs/{completed_job.id}/download.json" in markdown
```

- [ ] **Step 2: Run UI tests and verify failure**

Run: `pytest tests/test_ui.py -v`

Expected: FAIL because the history helpers do not exist.

- [ ] **Step 3: Replace request-bound transcription callbacks**

Change `create_ui(store, job_service)` so callbacks accept an injected
`request: gr.Request` and get `owner_id` from `request.request.state.owner_id`.
Remove `_stop_events`, `_session_tickets`, and task execution from `gr.State`.
The submit callback copies the Gradio upload into a persistent job and returns
its ID immediately.

- [ ] **Step 4: Add current-task polling and history controls**

Add a `gr.Dropdown` with `(label, job_id)` choices, refresh/open/delete buttons,
current status, transcript, JSON, copy-full-text and copy-text-only snapshot
buttons, and Markdown download links. A timer refreshes the selected/current job
and history. On initial `demo.load`, select the newest `queued`/`running` job,
otherwise the newest history item. The stop button calls
`store.request_cancel(owner_id, selected_job_id)`.

- [ ] **Step 5: Run UI tests**

Run: `pytest tests/test_ui.py tests/test_stop_button.py -v`

Expected: all tests PASS; legacy transcription parsing tests remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add app/ui.py tests/test_ui.py tests/test_stop_button.py
git commit -m "feat: restore private transcription history in UI"
```

### Task 9: Documentation, full verification, installation, and launch

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt` only if the installed FastAPI/Gradio versions prove incompatible with the implemented APIs.

- [ ] **Step 1: Document browser history behavior**

Add a README section explaining same-browser identity, loss of access after
cookie deletion, audio cleanup, persistent `data/`, TXT/JSON downloads, HTTPS
cookie configuration, and server-restart behavior.

- [ ] **Step 2: Run formatting/static sanity checks**

Run: `python -m compileall app tests`

Expected: exit code 0.

- [ ] **Step 3: Run focused security and persistence tests**

Run: `pytest tests/test_job_store.py tests/test_session.py tests/test_job_service.py tests/test_job_worker.py tests/test_job_api.py -v`

Expected: all tests PASS.

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md requirements.txt
git commit -m "docs: explain private persistent history"
```

- [ ] **Step 6: Install project dependencies**

Use the existing virtual environment when present; otherwise create `.venv`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Expected: dependency installation exits successfully. Model weights are not
downloaded during this step.

- [ ] **Step 7: Start the service for local testing**

Run in a persistent terminal session:

```bash
.venv/bin/python -m app.main
```

Expected: Uvicorn listens on the configured port and `/api/health` returns
`{"status":"ok"}`. Do not submit a real transcription during smoke testing,
because that loads/downloads the GPU model; leave the process running for the
user's browser test.

- [ ] **Step 8: Perform a cookie/history smoke test without the model**

Use one cookie jar to call health/list endpoints twice and a second cookie jar
to verify it receives a different session. Confirm `data/transcriber.sqlite3`
exists and no raw cookie token appears in it.

- [ ] **Step 9: Record final repository state**

Run: `git status --short && git log --oneline -10`

Expected: no unintended files are tracked; `data/` and `.venv/` remain ignored.
