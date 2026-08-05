# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
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

        # Check route behavior rather than FastAPI's private route container.
        response = TestClient(app).get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
