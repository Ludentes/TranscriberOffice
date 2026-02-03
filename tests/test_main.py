# tests/test_main.py
import pytest
from unittest.mock import Mock, patch


def test_create_app():
    """FastAPI app creates with routes mounted."""
    with patch('app.main.create_ui') as mock_ui, \
         patch('app.main.gr.mount_gradio_app') as mock_mount:
        mock_ui.return_value = Mock()
        # Make mount_gradio_app return the app unchanged
        mock_mount.side_effect = lambda app, ui, path: app

        from app.main import create_app

        app = create_app()

        # Check API routes are registered
        routes = [route.path for route in app.routes]
        assert "/api/transcribe" in routes or any("/api" in r for r in routes)
        assert "/api/health" in routes or any("health" in r for r in routes)
