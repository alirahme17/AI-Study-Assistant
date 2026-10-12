"""Per-user workspace helpers: paths, pipeline status, and the user's own API key.

Every user gets an isolated folder under ``workspaces/<username>/``:

    .env          GEMINI_API_KEY for this user only
    token.json    Google OAuth token for this user's Drive
    status.json   pipeline state: processing | ready | error
    data/         PDFs downloaded from Drive (one subfolder per course)
    chroma_db/    this user's vector database
"""
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from dotenv import dotenv_values

WORKSPACES_ROOT = "workspaces"

# Letters, digits, dot, dash, underscore; must not start with a dot.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_.\-]{0,63}$")

STATUS_NOT_CONFIGURED = "not_configured"
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_ERROR = "error"


def validate_username(username: str) -> str:
    """Reject usernames that could escape the workspaces folder (e.g. '../x')."""
    if not _USERNAME_RE.match(username or ""):
        raise ValueError(f"Invalid username: {username!r}")
    return username


def workspace_path(username: str) -> str:
    return os.path.join(WORKSPACES_ROOT, validate_username(username))


def legacy_db_path(username: str) -> str:
    """Where databases lived before versioned builds (still read if present)."""
    return os.path.join(workspace_path(username), "chroma_db")


def new_db_path(username: str) -> str:
    """A fresh, never-used folder for the next build (e.g. chroma_db_20261012T081500)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return os.path.join(workspace_path(username), f"chroma_db_{stamp}")


def db_path(username: str) -> Optional[str]:
    """The database that queries should use, or None if there is none yet.

    Each build writes to a brand-new folder and only becomes active once it has
    finished, so a half-built or failed build is never queried, and a rebuild
    never appends duplicate chunks to an old database.
    """
    status = read_status(username)
    if status.get("db_dir"):
        path = os.path.join(workspace_path(username), status["db_dir"])
        if os.path.isdir(path):
            return path
    legacy = legacy_db_path(username)
    return legacy if os.path.isdir(legacy) else None


def data_path(username: str) -> str:
    return os.path.join(workspace_path(username), "data")


def _status_file(username: str) -> str:
    return os.path.join(workspace_path(username), "status.json")


# ---------------------------------------------------------------- API key ---

def save_user_api_key(username: str, api_key: str) -> None:
    os.makedirs(workspace_path(username), exist_ok=True)
    with open(os.path.join(workspace_path(username), ".env"), "w") as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")


def get_user_api_key(username: str) -> Optional[str]:
    """Return this user's key, read straight from their own .env file.

    Deliberately does NOT touch os.environ: load_dotenv() never overwrites a
    variable that is already set, so the first user's key would otherwise be
    reused for every later user in the same server process.
    """
    env_file = os.path.join(workspace_path(username), ".env")
    if not os.path.exists(env_file):
        return None
    return dotenv_values(env_file).get("GEMINI_API_KEY") or None


# ----------------------------------------------------------------- status ---

def write_status(username: str, status: str, message: str = "",
                 db_dir: Optional[str] = None, embedding_model: Optional[str] = None) -> None:
    os.makedirs(workspace_path(username), exist_ok=True)
    previous = _read_status_file(username) or {}
    payload = {
        "status": status,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        # Keep pointing at the last good database until a new one is ready.
        "db_dir": db_dir if db_dir is not None else previous.get("db_dir"),
        # Which embedding model built that database (queries must use the same one).
        "embedding_model": embedding_model if embedding_model is not None else previous.get("embedding_model"),
    }
    tmp = _status_file(username) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, _status_file(username))  # atomic: readers never see half a file


def _read_status_file(username: str) -> Optional[dict]:
    try:
        with open(_status_file(username)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_status(username: str) -> dict:
    status = _read_status_file(username)
    if status is not None:
        return status
    # Workspaces built before status.json existed: a DB folder means it finished.
    if os.path.isdir(legacy_db_path(username)):
        return {"status": STATUS_READY, "message": ""}
    return {"status": STATUS_NOT_CONFIGURED, "message": ""}


def embedding_model_for(username: str) -> str:
    """The embedding model the user's active database was built with."""
    from embeddings import LEGACY_MODEL
    return read_status(username).get("embedding_model") or LEGACY_MODEL
