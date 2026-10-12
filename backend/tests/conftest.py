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
