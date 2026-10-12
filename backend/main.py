import json
import os
import threading
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from build_database import EMBEDDING_MODEL, build_user_database, remove_old_databases
from ingest import extract_folder_id, ingest_drive_data
from rag import build_messages, condense_question, message_text, number_sources, trim_history
from workspace import (
    STATUS_ERROR,
    STATUS_PROCESSING,
    STATUS_READY,
    data_path,
    db_path,
    get_user_api_key,
    read_status,
    save_user_api_key,
    validate_username,
    write_status,
)

app = FastAPI()

# Gemini 1.5/2.0 models are shut down. Override with the GEMINI_MODEL env var,
# e.g. GEMINI_MODEL=gemini-3.5-flash-lite for a cheaper, faster model.
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
TOP_K = 5

# Pipelines running in THIS server process. If status.json says "processing"
# but the user isn't in here, the server was restarted mid-build and that
# build is dead — we report it as an error instead of "processing" forever.
_running: set = set()
_running_lock = threading.Lock()


class ConfigurePayload(BaseModel):
    username: str
    drive_folder_url: str
    gemini_api_key: str


class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class QueryPayload(BaseModel):
    username: str
    question: str
    course: Optional[str] = None
    history: List[ChatMessage] = []   # previous turns of this chat, oldest first


def _checked_username(username: str) -> str:
    try:
        return validate_username(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once per process instead of on every question."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def open_vector_store(persist_directory: str):
    from langchain_community.vectorstores import Chroma
    return Chroma(persist_directory=persist_directory, embedding_function=get_embeddings())


def make_llm(api_key: str, model: Optional[str] = None):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model or LLM_MODEL, google_api_key=api_key)


def run_data_pipeline(username: str, drive_folder_id: Optional[str]):
    """Long-running ingestion + indexing, run in a background thread.

    drive_folder_id=None re-indexes the PDFs already downloaded (no Drive access).
    Always ends in either "ready" or "error" — never stays "processing".
    """
    try:
        if drive_folder_id is not None:
            write_status(username, STATUS_PROCESSING, "Downloading PDFs from Google Drive...")
            pdf_count = ingest_drive_data(username, drive_folder_id)
            if pdf_count == 0:
                raise RuntimeError("No PDFs were found in that Google Drive folder.")
            label = f"{pdf_count} PDF(s)"
        else:
            label = "your downloaded PDFs"

        write_status(username, STATUS_PROCESSING, f"Indexing {label}...")
        new_db_dir = build_user_database(username)

        write_status(username, STATUS_READY, f"Indexed {label}.", db_dir=new_db_dir)
        remove_old_databases(username, keep_dir=new_db_dir)
        print(f"Data pipeline for user '{username}' completed successfully.")
    except Exception as e:
        print(f"Error in data pipeline for user '{username}': {e}")
        write_status(username, STATUS_ERROR, f"{type(e).__name__}: {e}")
    finally:
        with _running_lock:
            _running.discard(username)


def _start_pipeline(username: str, drive_folder_id: Optional[str]) -> None:
    with _running_lock:
        if username in _running:
            raise HTTPException(status_code=409, detail="Your knowledge base is already being built.")
        _running.add(username)
    try:
        write_status(username, STATUS_PROCESSING, "Starting...")
        threading.Thread(
            target=run_data_pipeline, args=(username, drive_folder_id), daemon=True
        ).start()
    except Exception:
        with _running_lock:
            _running.discard(username)
        raise


@app.post("/api/configure")
def configure_workspace(payload: ConfigurePayload):
    username = _checked_username(payload.username)

    drive_folder_id = extract_folder_id(payload.drive_folder_url)
    if not drive_folder_id:
        raise HTTPException(status_code=400, detail="Could not find a folder ID in that Google Drive link.")
    if not payload.gemini_api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key is empty.")

    save_user_api_key(username, payload.gemini_api_key.strip())
    _start_pipeline(username, drive_folder_id)
    return {"message": f"Configuration received for '{username}'. Data processing started."}


@app.post("/api/rebuild/{username}")
def rebuild_index(username: str):
    """Re-index the PDFs already downloaded, e.g. after an indexing upgrade (page numbers)."""
    username = _checked_username(username)
    if not os.path.isdir(data_path(username)):
        raise HTTPException(status_code=404, detail="No downloaded PDFs found. Please run setup first.")
    _start_pipeline(username, None)
    return {"message": "Re-indexing started."}


@app.get("/api/status/{username}")
def get_status(username: str):
    username = _checked_username(username)
    status = read_status(username)
    if status.get("status") == STATUS_PROCESSING:
        with _running_lock:
            alive = username in _running
        if not alive:
            status = {
                "status": STATUS_ERROR,
                "message": "Processing was interrupted (the server restarted). Please run setup again.",
            }
    return {"status": status.get("status"), "message": status.get("message", "")}


@app.get("/api/courses/{username}")
def get_courses(username: str):
    data_dir = data_path(_checked_username(username))
    try:
        courses = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        return {"courses": sorted(courses)}
    except FileNotFoundError:
        return {"courses": []}


def retrieve(llm, db, question: str, course: Optional[str], history: list, k: int = TOP_K):
    """The retrieval half of the RAG pipeline, shared by the API and the evaluation.

    Returns (search_query, docs). Memory step 1: with chat history, follow-ups like
    "what about its range?" are rewritten into a standalone search query first.
    """
    search_query = condense_question(llm, question, history)
    search_kwargs = {"k": k}
    if course:
        search_kwargs["filter"] = {"course": {"$eq": course}}
    return search_query, db.similarity_search(query=search_query, **search_kwargs)


def _event(kind: str, **data) -> str:
    """One line of newline-delimited JSON (NDJSON)."""
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


@app.post("/api/query")
def query_database(payload: QueryPayload):
    """Answer a question with citations, streamed as NDJSON events:

        {"type": "sources", "sources": [{"id": 1, "source": ..., "page": 3, ...}, ...]}
        {"type": "token",   "text": "Radar range depends on [1] ..."}   (many)
        {"type": "error",   "message": "..."}                          (only on failure)
        {"type": "done"}
    """
    username = _checked_username(payload.username)

    persist_directory = db_path(username)
    if persist_directory is None:
        raise HTTPException(status_code=404, detail="Database not found for this user. Please configure first.")

    # Each user's own key, read from their own .env — never shared via os.environ.
    api_key = get_user_api_key(username)
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key not found for this user.")

    history = trim_history([m.model_dump() for m in payload.history])

    def stream_events():
        try:
            llm = make_llm(api_key)
            db = open_vector_store(persist_directory)

            search_query, docs = retrieve(llm, db, payload.question, payload.course, history)
            context, sources = number_sources(docs)
            yield _event("sources", sources=sources, search_query=search_query)

            # Memory, step 2: the model sees the recent conversation too.
            for chunk in llm.stream(build_messages(payload.question, context, history)):
                text = message_text(chunk)
                if text:
                    yield _event("token", text=text)
        except Exception as e:
            yield _event("error", message=f"{type(e).__name__}: {e}")
        yield _event("done")

    return StreamingResponse(stream_events(), media_type="application/x-ndjson")
