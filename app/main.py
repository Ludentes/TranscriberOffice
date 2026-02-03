# app/main.py
"""Main application entry point."""
import gradio as gr
from fastapi import FastAPI

from app.api import router as api_router
from app.config import get_config
from app.ui import create_ui


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Meeting Transcriber",
        description="Transcribe meetings with speaker identification using VibeVoice-ASR",
        version="1.0.0"
    )

    # Include API routes
    app.include_router(api_router)

    # Mount Gradio UI
    ui = create_ui()
    app = gr.mount_gradio_app(app, ui, path="/")

    return app


def main():
    """Run the application."""
    import uvicorn

    config = get_config()

    print(f"Starting Meeting Transcriber on {config.server.host}:{config.server.port}")
    print(f"Web UI: http://{config.server.host}:{config.server.port}/")
    print(f"API: http://{config.server.host}:{config.server.port}/api/transcribe")

    uvicorn.run(
        create_app(),
        host=config.server.host,
        port=config.server.port
    )


if __name__ == "__main__":
    main()
