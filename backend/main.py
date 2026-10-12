import json
import os
import threading
from functools import lru_cache
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

import auth
from auth import current_user

from build_database import build_user_database, remove_old_databases
from embeddings import DEFAULT_MODEL as EMBEDDING_MODEL, LEGACY_MODEL, get_embeddings
from ingest import extract_folder_id, ingest_drive_data, safe_filename
from pages import page_count, render_page, resolve_source
from rag import build_messages, condense_question, message_text, number_sources, trim_history
from search import KeywordIndex, search
from workspace import (
    STATUS_ERROR,
    STATUS_PROCESSING,
    STATUS_READY,
    data_path,
    db_path,
    embedding_model_for,
    get_user_api_key,
    read_status,
    save_user_api_key,
    validate_username,
    write_status,
)

app = FastAPI(title="Teleco-Tutor API")

# The Next.js frontend runs on another origin (localhost:3000 in dev, Vercel in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

FAKE_LLM = os.getenv("FAKE_LLM") == "1"          # offline demo mode, see devllm.py
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Gemini 1.5/2.0 models are shut down. Override with the GEMINI_MODEL env var,
# e.g. GEMINI_MODEL=gemini-3.5-flash-lite for a cheaper, faster model.
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
TOP_K = 5
# "hybrid" (keyword BM25 + vector, fused), "vector" (embeddings only) or "bm25" (keywords only).
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

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


def legacy_routes_allowed():
    """The original username-in-the-URL routes (used by the Streamlit app) have no
    authentication, so they only exist in local dev mode."""
    if auth.AUTH_MODE != "dev":
        raise HTTPException(status_code=404, detail="Not found.")


def _checked_username(username: str) -> str:
    try:
        return validate_username(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def open_vector_store(persist_directory: str, embedding_model: str = LEGACY_MODEL):
    """Open a database with the embedding model it was built with (see embedding_model_for)."""
    from langchain_community.vectorstores import Chroma
    return Chroma(persist_directory=persist_directory, embedding_function=get_embeddings(embedding_model))


@lru_cache(maxsize=8)
def get_keyword_index(persist_directory: str) -> KeywordIndex:
    """BM25 index over a database's chunks, built once per database folder (a rebuild
    creates a new folder, so the cache never serves a stale index)."""
    return KeywordIndex.from_chroma(open_vector_store(persist_directory))


def safe_keyword_index(persist_directory: str) -> Optional[KeywordIndex]:
    """Keyword index, or None (-> vector-only search) if it cannot be built."""
    if RETRIEVAL_MODE == "vector":
        return None
    try:
        return get_keyword_index(persist_directory)
    except Exception as e:
        print(f"Keyword index unavailable, using vector search only: {type(e).__name__}: {e}")
        return None


def make_llm(api_key: str, model: Optional[str] = None):
    if FAKE_LLM:
        from devllm import FakeChatModel
        return FakeChatModel()
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
        new_db_dir = build_user_database(
            username, progress=lambda message: write_status(username, STATUS_PROCESSING, message),
            embedding_model=EMBEDDING_MODEL)

        write_status(username, STATUS_READY, f"Indexed {label}.", db_dir=new_db_dir,
                     embedding_model=EMBEDDING_MODEL)
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


@app.post("/api/configure", dependencies=[Depends(legacy_routes_allowed)])
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


@app.post("/api/rebuild/{username}", dependencies=[Depends(legacy_routes_allowed)])
def rebuild_index(username: str):
    """Re-index the PDFs already downloaded, e.g. after an indexing upgrade (page numbers)."""
    username = _checked_username(username)
    if not os.path.isdir(data_path(username)):
        raise HTTPException(status_code=404, detail="No downloaded PDFs found. Please run setup first.")
    _start_pipeline(username, None)
    return {"message": "Re-indexing started."}


def live_status(username: str) -> dict:
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


@app.get("/api/status/{username}", dependencies=[Depends(legacy_routes_allowed)])
def get_status(username: str):
    return live_status(_checked_username(username))


def list_courses(username: str) -> List[str]:
    data_dir = data_path(username)
    try:
        return sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    except FileNotFoundError:
        return []


@app.get("/api/courses/{username}", dependencies=[Depends(legacy_routes_allowed)])
def get_courses(username: str):
    return {"courses": list_courses(_checked_username(username))}


def retrieve(llm, db, question: str, course: Optional[str], history: list, k: int = TOP_K,
             keyword_index: Optional[KeywordIndex] = None, mode: Optional[str] = None):
    """The retrieval half of the RAG pipeline, shared by the API and the evaluation.

    Returns (search_query, docs). Memory step 1: with chat history, follow-ups like
    "what about its range?" are rewritten into a standalone search query first.
    Then hybrid search: keyword (BM25) and vector results fused with RRF.
    """
    search_query = condense_question(llm, question, history)
    docs = search(db, keyword_index, search_query, k, course or None, mode or RETRIEVAL_MODE)
    return search_query, docs


def _event(kind: str, **data) -> str:
    """One line of newline-delimited JSON (NDJSON)."""
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


def answer_stream(username: str, question: str, course: Optional[str], history_messages: list):
    """Answer a question with citations, streamed as NDJSON events:

        {"type": "sources", "sources": [{"id": 1, "source": ..., "page": 3, ...}, ...]}
        {"type": "token",   "text": "Radar range depends on [1] ..."}   (many)
        {"type": "error",   "message": "..."}                          (only on failure)
        {"type": "done"}
    """
    persist_directory = db_path(username)
    if persist_directory is None:
        raise HTTPException(status_code=404, detail="You have no knowledge base yet. Add PDFs and index them first.")

    # Each user's own key, read from their own .env — never shared via os.environ.
    api_key = get_user_api_key(username)
    if not api_key and not FAKE_LLM:
        raise HTTPException(status_code=400, detail="Add your Gemini API key in Settings first.")

    history = trim_history(history_messages)

    def stream_events():
        try:
            llm = make_llm(api_key)
            db = open_vector_store(persist_directory, embedding_model_for(username))

            search_query, docs = retrieve(llm, db, question, course, history,
                                          keyword_index=safe_keyword_index(persist_directory))
            context, sources = number_sources(docs)
            yield _event("sources", sources=sources, search_query=search_query)

            # Memory, step 2: the model sees the recent conversation too.
            for chunk in llm.stream(build_messages(question, context, history)):
                text = message_text(chunk)
                if text:
                    yield _event("token", text=text)
        except Exception as e:
            yield _event("error", message=f"{type(e).__name__}: {e}")
        yield _event("done")

    return StreamingResponse(stream_events(), media_type="application/x-ndjson")


@app.post("/api/query", dependencies=[Depends(legacy_routes_allowed)])
def query_database(payload: QueryPayload):
    username = _checked_username(payload.username)
    return answer_stream(username, payload.question, payload.course, [m.model_dump() for m in payload.history])


# =====================================================================================
# Authenticated API for the Next.js frontend. The user always comes from the sign-in
# token (see auth.py), never from the request body or URL.
# =====================================================================================

class ChatPayload(BaseModel):
    question: str
    course: Optional[str] = None
    history: List[ChatMessage] = []


class KeyPayload(BaseModel):
    gemini_api_key: str


class DrivePayload(BaseModel):
    drive_folder_url: str


def course_files(username: str) -> List[dict]:
    """[{"name": "Radar", "files": [{"source": "Radar/Course/c1.pdf", "name": "c1.pdf"}, ...]}, ...]"""
    root = data_path(username)
    courses = []
    for course in list_courses(username):
        files = []
        for dirpath, _dirs, names in os.walk(os.path.join(root, course)):
            for name in sorted(names):
                if name.lower().endswith(".pdf"):
                    rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
                    files.append({"source": rel, "name": name})
        courses.append({"name": course, "files": sorted(files, key=lambda f: f["source"])})
    return courses


@app.get("/api/health")
def health():
    return {"ok": True, "auth_mode": auth.AUTH_MODE, "fake_llm": FAKE_LLM}


@app.get("/api/me")
def me(username: str = Depends(current_user)):
    status = live_status(username)
    return {
        "workspace": username,
        "status": status["status"],
        "message": status["message"],
        "has_api_key": bool(get_user_api_key(username)) or FAKE_LLM,
        "embedding_model": embedding_model_for(username) if db_path(username) else None,
        "courses": course_files(username),
    }


@app.put("/api/me/key")
def set_key(payload: KeyPayload, username: str = Depends(current_user)):
    key = payload.gemini_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="The API key is empty.")
    save_user_api_key(username, key)
    return {"saved": True}


@app.post("/api/me/upload")
async def upload_pdfs(course: str = Form(...), files: List[UploadFile] = File(...),
                      username: str = Depends(current_user)):
    """Save uploaded PDFs under data/<course>/. Index afterwards with POST /api/me/index."""
    course_dir_name = safe_filename(course.strip())
    if not course.strip():
        raise HTTPException(status_code=400, detail="Choose a course name.")
    target = os.path.join(data_path(username), course_dir_name)
    os.makedirs(target, exist_ok=True)
    saved = []
    for upload in files:
        name = safe_filename(os.path.basename(upload.filename or "document.pdf"))
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload.filename}: only PDF files can be uploaded.")
        data = await upload.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{upload.filename} is larger than {MAX_UPLOAD_MB} MB.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail=f"{upload.filename} is not a valid PDF.")
        with open(os.path.join(target, name), "wb") as f:
            f.write(data)
        saved.append(f"{course_dir_name}/{name}")
    return {"saved": saved}


@app.delete("/api/me/files")
def delete_pdf(source: str = Query(...), username: str = Depends(current_user)):
    try:
        os.remove(resolve_source(username, source))
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="File not found.")
    return {"deleted": source}


@app.post("/api/me/index")
def index_pdfs(username: str = Depends(current_user)):
    """(Re)build the knowledge base from the user's PDFs."""
    if not any(c["files"] for c in course_files(username)):
        raise HTTPException(status_code=400, detail="Add some PDFs first.")
    _start_pipeline(username, None)
    return {"message": "Indexing started."}


@app.post("/api/me/drive")
def import_drive(payload: DrivePayload, username: str = Depends(current_user)):
    """Import PDFs from a Google Drive folder, then index. (Local use: the Google sign-in
    currently opens on the machine running the backend.)"""
    folder_id = extract_folder_id(payload.drive_folder_url)
    if not folder_id:
        raise HTTPException(status_code=400, detail="Could not find a folder ID in that Google Drive link.")
    _start_pipeline(username, folder_id)
    return {"message": "Import started."}


@app.get("/api/me/page")
def page_image(source: str = Query(...), page: int = Query(..., ge=1), highlight: Optional[str] = Query(None),
               username: str = Depends(current_user)):
    """PNG of one page of the user's PDF, with the cited passage highlighted."""
    try:
        path = resolve_source(username, source)
        png = render_page(path, page, highlight)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="File not found.")
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(png, media_type="image/png", headers={
        "Cache-Control": "private, max-age=3600",
        "X-Page-Count": str(page_count(path)),
        "Access-Control-Expose-Headers": "X-Page-Count",
    })


@app.post("/api/chat")
def chat(payload: ChatPayload, username: str = Depends(current_user)):
    return answer_stream(username, payload.question, payload.course, [m.model_dump() for m in payload.history])
