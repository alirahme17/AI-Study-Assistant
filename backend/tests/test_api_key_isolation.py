"""Bug 1: the first user's Gemini key was reused for every later user."""
from fastapi.testclient import TestClient

import main
from workspace import save_user_api_key, write_status, STATUS_READY


class FakeStore:
    def similarity_search(self, query, **kwargs):
        return []


class FakeLLM:
    def __init__(self, key):
        self.key = key

    def stream(self, messages):
        class Chunk:
            content = f"answered-with:{self.key}"
        yield Chunk()


def answer_text(response) -> str:
    import json
    events = [json.loads(line) for line in response.text.splitlines() if line]
    return "".join(e["text"] for e in events if e["type"] == "token")


def _ready_user(tmp_path, username, key):
    save_user_api_key(username, key)
    db_dir = tmp_path / "workspaces" / username / "chroma_db_test"
    db_dir.mkdir(parents=True)
    write_status(username, STATUS_READY, db_dir="chroma_db_test")


def test_each_user_gets_their_own_key(isolated_workspaces, monkeypatch):
    monkeypatch.setattr(main, "open_vector_store", lambda path: FakeStore())
    monkeypatch.setattr(main, "make_llm", lambda key: FakeLLM(key))
    _ready_user(isolated_workspaces, "alice", "KEY-ALICE")
    _ready_user(isolated_workspaces, "bob", "KEY-BOB")

    client = TestClient(main.app)
    first = client.post("/api/query", json={"username": "alice", "question": "q"})
    second = client.post("/api/query", json={"username": "bob", "question": "q"})

    assert answer_text(first) == "answered-with:KEY-ALICE"
    assert answer_text(second) == "answered-with:KEY-BOB"  # was KEY-ALICE before the fix


def test_key_is_not_leaked_into_process_environment(isolated_workspaces, monkeypatch):
    import os
    from workspace import get_user_api_key
    save_user_api_key("alice", "KEY-ALICE")
    assert get_user_api_key("alice") == "KEY-ALICE"
    assert "GEMINI_API_KEY" not in os.environ


def test_path_traversal_username_rejected():
    client = TestClient(main.app)
    assert client.get("/api/status/..").status_code in (400, 404)
    assert client.post("/api/query", json={"username": "../other", "question": "q"}).status_code == 400
