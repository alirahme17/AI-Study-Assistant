import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)


@pytest.fixture(autouse=True)
def isolated_workspaces(tmp_path, monkeypatch):
    """Run every test in an empty temp folder, so real workspaces are never touched."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield tmp_path
    # Let background pipeline threads finish before the next test changes folder,
    # otherwise a slow thread could write its status into the next test's workspace.
    import time
    import main
    deadline = time.time() + 5
    while main._running and time.time() < deadline:
        time.sleep(0.01)
