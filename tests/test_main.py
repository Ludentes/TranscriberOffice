# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.config import load_config


def test_create_app(tmp_path):
    """FastAPI app creates with routes mounted."""
    with patch('app.main.create_ui') as mock_ui, \
         patch('app.main.gr.mount_gradio_app') as mock_mount, \
         patch('app.main.JobWorker') as worker_class:
        mock_ui.return_value = Mock()
        # Make mount_gradio_app return the app unchanged
        mock_mount.side_effect = lambda app, ui, path: app
        config = load_config(tmp_path / "missing.yaml")
        config.storage.data_dir = str(tmp_path / "data")

        from app.main import create_app

        app = create_app(config)

        # Check route behavior rather than FastAPI's private route container.
        with TestClient(app) as client:
            response = client.get("/api/health")
            history = client.get("/api/jobs")
            token_export = client.get("/api/session/token")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            assert history.status_code == 200
            assert history.json() == []
            assert "transcriber_session=" in response.headers["set-cookie"]
            assert token_export.status_code == 200
            assert len(token_export.json()["token"]) == 64
            assert token_export.headers["cache-control"] == "no-store"

        worker = worker_class.return_value
        worker.start.assert_called_once_with()
        worker.stop.assert_called_once_with()
        assert app.state.job_store.db_path == tmp_path / "data" / "transcriber.sqlite3"
        mock_ui.assert_called_once_with(app.state.job_store, app.state.job_service)
