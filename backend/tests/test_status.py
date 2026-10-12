"""Bug 2: setup could show "processing" forever (failed build, half-built DB, server restart)."""
import time

import pytest
from fastapi.testclient import TestClient

import main
import workspace


def _wait_until_not_processing(client, username, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/status/{username}").json()
        if body["status"] != "processing":
            return body
        time.sleep(0.05)
    pytest.fail("status stayed 'processing'")


def _configure(client, username="alice"):
    return client.post("/api/configure", json={
        "username": username,
        "drive_folder_url": "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp?usp=sharing",
        "gemini_api_key": "KEY",
    })


def test_unconfigured_user_is_not_reported_as_processing():
    client = TestClient(main.app)
    assert client.get("/api/status/newuser").json()["status"] == "not_configured"


def test_failed_ingestion_ends_in_error(monkeypatch):
    def boom(username, folder_id):
        raise RuntimeError("Drive said no")
    monkeypatch.setattr(main, "ingest_drive_data", boom)

    client = TestClient(main.app)
    assert _configure(client).status_code == 200
    body = _wait_until_not_processing(client, "alice")
    assert body["status"] == "error"
    assert "Drive said no" in body["message"]


def test_empty_folder_ends_in_error(monkeypatch):
    monkeypatch.setattr(main, "ingest_drive_data", lambda u, f: 0)
    client = TestClient(main.app)
    _configure(client)
    body = _wait_until_not_processing(client, "alice")
    assert body["status"] == "error"
    assert "No PDFs" in body["message"]


def test_success_ends_in_ready_and_points_at_new_db(isolated_workspaces, monkeypatch):
    monkeypatch.setattr(main, "ingest_drive_data", lambda u, f: 3)

    def fake_build(username, **kwargs):
        (isolated_workspaces / "workspaces" / username / "chroma_db_new").mkdir(parents=True)
        return "chroma_db_new"
    monkeypatch.setattr(main, "build_user_database", fake_build)

    client = TestClient(main.app)
    _configure(client)
    body = _wait_until_not_processing(client, "alice")
    assert body["status"] == "ready"
    assert workspace.db_path("alice").endswith("chroma_db_new")


def test_half_built_db_is_not_reported_ready(isolated_workspaces, monkeypatch):
    """Old code said "ready" as soon as the DB folder existed, even mid-build."""
    started = []

    def slow_build(username, **kwargs):
        (isolated_workspaces / "workspaces" / username / "chroma_db_partial").mkdir(parents=True)
        started.append(True)
        time.sleep(0.5)
        raise RuntimeError("crashed mid-build")
    monkeypatch.setattr(main, "ingest_drive_data", lambda u, f: 1)
    monkeypatch.setattr(main, "build_user_database", slow_build)

    client = TestClient(main.app)
    _configure(client)
    deadline = time.time() + 5
    while not started:
        assert time.time() < deadline, "build never started"
        time.sleep(0.01)
    assert client.get("/api/status/alice").json()["status"] == "processing"
    assert _wait_until_not_processing(client, "alice")["status"] == "error"
    assert workspace.db_path("alice") is None


def test_server_restart_mid_build_is_reported_as_error():
    workspace.write_status("alice", workspace.STATUS_PROCESSING, "Indexing...")
    # Nothing is running in this process -> the build died with the old server.
    body = TestClient(main.app).get("/api/status/alice").json()
    assert body["status"] == "error"
    assert "interrupted" in body["message"]


def test_double_configure_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "ingest_drive_data", lambda u, f: time.sleep(0.5) or 0)
    client = TestClient(main.app)
    assert _configure(client).status_code == 200
    assert _configure(client).status_code == 409
    _wait_until_not_processing(client, "alice")


def test_old_workspaces_without_status_file_still_work(isolated_workspaces):
    (isolated_workspaces / "workspaces" / "legacy" / "chroma_db").mkdir(parents=True)
    assert TestClient(main.app).get("/api/status/legacy").json()["status"] == "ready"
    assert workspace.db_path("legacy").endswith("chroma_db")
