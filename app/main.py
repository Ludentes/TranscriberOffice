# app/main.py
"""Main application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI

from app.api import router as api_router
from app.config import AppConfig, get_config
from app.job_api import create_job_router
from app.job_service import JobService
from app.job_store import JobStore
from app.job_worker import JobWorker
from app.session import BrowserSessionMiddleware
from app.transcribe import get_transcription_service
from app.ui import create_ui


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or get_config()
    data_dir = Path(config.storage.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(data_dir / "transcriber.sqlite3")
    job_service = JobService(
        store,
        data_dir,
        max_file_size_mb=config.transcription.max_file_size_mb,
    )
    worker = JobWorker(store, job_service, get_transcription_service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(
        title="Meeting Transcriber",
        description="Transcribe meetings with speaker identification using VibeVoice-ASR",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.job_store = store
    app.state.job_service = job_service
    app.state.job_worker = worker
    app.add_middleware(
        BrowserSessionMiddleware,
        store=store,
        settings=config.session,
    )

    # Include API routes
    app.include_router(api_router)
    app.include_router(create_job_router(store, job_service))

    # Mount Gradio UI
    ui = create_ui(store, job_service)
    app = gr.mount_gradio_app(app, ui, path="/")

    return app


def main():
    """Run the application."""
    try:
        config = get_config()

        print(f"Starting Meeting Transcriber on {config.server.host}:{config.server.port}")
        print(f"Web UI: http://{config.server.host}:{config.server.port}/")
        print(f"API: http://{config.server.host}:{config.server.port}/api/transcribe")

        uvicorn.run(
            create_app(config),
            host=config.server.host,
            port=config.server.port
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
