import os
import threading
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from build_database import EMBEDDING_MODEL, build_user_database, remove_old_databases
from ingest import extract_folder_id, ingest_drive_data
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

LLM_MODEL = "gemini-1.5-flash"

# Pipelines running in THIS server process. If status.json says "processing"
# but the user isn't in here, the server was restarted mid-build and that
# build is dead — we report it as an error instead of "processing" forever.
_running: set = set()
_running_lock = threading.Lock()


class ConfigurePayload(BaseModel):
    username: str
    drive_folder_url: str
    gemini_api_key: str


class QueryPayload(BaseModel):
    username: str
    question: str
    course: Optional[str] = None


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


def make_llm(api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key)


def run_data_pipeline(username: str, drive_folder_id: str):
    """Long-running ingestion + indexing, run in a background thread.

    Always ends in either "ready" or "error" — never stays "processing".
    """
    try:
        write_status(username, STATUS_PROCESSING, "Downloading PDFs from Google Drive...")
        pdf_count = ingest_drive_data(username, drive_folder_id)
        if pdf_count == 0:
            raise RuntimeError("No PDFs were found in that Google Drive folder.")

        write_status(username, STATUS_PROCESSING, f"Indexing {pdf_count} PDF(s)...")
        new_db_dir = build_user_database(username)

        write_status(username, STATUS_READY, f"Indexed {pdf_count} PDF(s).", db_dir=new_db_dir)
        remove_old_databases(username, keep_dir=new_db_dir)
        print(f"Data pipeline for user '{username}' completed successfully.")
    except Exception as e:
        print(f"Error in data pipeline for user '{username}': {e}")
        write_status(username, STATUS_ERROR, f"{type(e).__name__}: {e}")
    finally:
        with _running_lock:
            _running.discard(username)


@app.post("/api/configure")
def configure_workspace(payload: ConfigurePayload):
    username = _checked_username(payload.username)

    drive_folder_id = extract_folder_id(payload.drive_folder_url)
    if not drive_folder_id:
        raise HTTPException(status_code=400, detail="Could not find a folder ID in that Google Drive link.")
    if not payload.gemini_api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key is empty.")

    with _running_lock:
        if username in _running:
            raise HTTPException(status_code=409, detail="Your knowledge base is already being built.")
        _running.add(username)

    try:
        save_user_api_key(username, payload.gemini_api_key.strip())
        write_status(username, STATUS_PROCESSING, "Starting...")
        threading.Thread(
            target=run_data_pipeline, args=(username, drive_folder_id), daemon=True
        ).start()
    except Exception:
        with _running_lock:
            _running.discard(username)
        raise
    return {"message": f"Configuration received for '{username}'. Data processing started."}


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


@app.post("/api/query")
def query_database(payload: QueryPayload):
    username = _checked_username(payload.username)

    persist_directory = db_path(username)
    if persist_directory is None:
        raise HTTPException(status_code=404, detail="Database not found for this user. Please configure first.")

    # Each user's own key, read from their own .env — never shared via os.environ.
    api_key = get_user_api_key(username)
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key not found for this user.")

    db = open_vector_store(persist_directory)
    llm = make_llm(api_key)

    search_kwargs = {"k": 4}
    if payload.course:
        search_kwargs["filter"] = {"course": {"$eq": payload.course}}

    retrieved_docs = db.similarity_search(query=payload.question, **search_kwargs)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt_template = f"Answer based only on the context provided.\n\nContext:\n{context}\n\nQuestion:\n{payload.question}"

    def stream_generator():
        try:
            for chunk in llm.stream(prompt_template):
                yield chunk.content
        except Exception as e:
            yield f"Error during response generation: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")
