"""Test setup: a real Postgres with pgvector, offline AI providers, local file storage.

Set TEST_DATABASE_URL to point at a throwaway database (default: localhost:5433/tutor_test).
Every table is emptied before each test. Tests that need the database are skipped
when it isn't reachable.
"""
import os
import sys
import tempfile
import time

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

# Must be set before settings.py is imported.
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL", "postgresql://postgres@localhost:5433/tutor_test")
os.environ["EMBEDDINGS"] = "fake"
os.environ["OCR"] = "none"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = tempfile.mkdtemp(prefix="tt-storage-")
os.environ.pop("SUPABASE_URL", None)
os.environ.pop("ENCRYPTION_KEY", None)
os.environ.pop("FAKE_LLM", None)

_db_state = {"checked": False, "error": None}


def _database():
    if not _db_state["checked"]:
        _db_state["checked"] = True
        try:
            import db
            with db.connect() as conn:
                db.apply_schema(conn)
        except Exception as e:
            _db_state["error"] = f"{type(e).__name__}: {e}"
    return _db_state["error"]


def wait_for_indexing(user_id=None, timeout=20):
    """Block until no indexing worker is running (for user_id, or for anyone)."""
    import indexer
    deadline = time.time() + timeout
    while time.time() < deadline:
        with indexer._lock:
            busy = (user_id in indexer._running) if user_id else bool(indexer._running)
        if not busy:
            return
        time.sleep(0.02)
    raise AssertionError("indexing did not finish")


@pytest.fixture
def database():
    """A clean database (all tables emptied)."""
    error = _database()
    if error:
        pytest.skip(f"test database unavailable: {error}")
    import db
    import indexer
    wait_for_indexing()
    indexer._paused_until.clear()
    with db.connect() as conn:
        conn.execute("truncate tt_chunks, tt_documents, tt_profiles, tt_embedding_cache, tt_ocr_cache")
    yield
    wait_for_indexing()


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield tmp_path
