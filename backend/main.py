"""Teleco-Tutor API.

    browser (Next.js) --Supabase sign-in token--> this API --> Postgres (pgvector + full text)
                                                          --> Storage (the PDFs)
                                                          --> Gemini (embeddings, OCR, answers)

Every route takes the user from the sign-in token (auth.current_user), never from
the request body or URL, so a user can only reach their own documents.
"""
import json
import time
import traceback
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

import auth
import db
import documents
import indexer
import settings
from auth import current_user
from pages import render_page
from providers import make_embedder
from rag import build_messages, condense_question, message_text, number_sources, trim_history
from search import search
from storage import get_storage

_startup = {"db": "not checked"}


def prepare_backend() -> None:
    """Create the tables and the storage bucket if they don't exist yet (idempotent)."""
    try:
        with db.connect() as conn:
            if settings.AUTO_MIGRATE:
                db.apply_schema(conn)
            conn.execute("select 1 from tt_documents limit 1")
        get_storage().ensure_ready()
        _startup["db"] = "ok"
    except Exception as e:
        traceback.print_exc()
        _startup["db"] = f"{type(e).__name__}: {e}"[:300]


@asynccontextmanager
async def lifespan(_app):
    prepare_backend()
    yield


app = FastAPI(title="Teleco-Tutor API", lifespan=lifespan)

# The Next.js frontend runs on another origin (localhost:3000 in dev, Vercel in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Page-Count"],
)


class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatPayload(BaseModel):
    question: str
    course: Optional[str] = None
    history: List[ChatMessage] = []   # previous turns of this chat, oldest first


class KeyPayload(BaseModel):
    gemini_api_key: str


def make_llm(api_key: str, model: Optional[str] = None):
    if settings.FAKE_LLM:
        from devllm import FakeChatModel
        return FakeChatModel()
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model or settings.GEMINI_MODEL, google_api_key=api_key)


def retrieve(llm, conn, user_id, embedder, question: str, course: Optional[str], history: list,
             k: int = settings.TOP_K, mode: Optional[str] = None):
    """The retrieval half of the RAG pipeline, shared by the API and the evaluation.

    Returns (search_query, docs). Memory step 1: with chat history, follow-ups like
    "what about its range?" are rewritten into a standalone search query first.
    """
    search_query = condense_question(llm, question, history)
    docs = search(conn, user_id, embedder, search_query, k, course or None, mode or settings.RETRIEVAL_MODE)
    return search_query, docs


def _event(kind: str, **data) -> str:
    """One line of newline-delimited JSON (NDJSON)."""
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


def _friendly(e: Exception) -> str:
    from quota import QuotaExhausted
    if isinstance(e, QuotaExhausted):
        return "Your Gemini API key reached its free daily limit. Try again tomorrow, or use another key."
    return f"{type(e).__name__}: {e}"


def answer_stream(user_id: str, question: str, course: Optional[str], history_messages: list):
    """Answer a question with citations, streamed as NDJSON events:

        {"type": "sources", "sources": [{"id": 1, "source": ..., "page": 3, ...}, ...]}
        {"type": "token",   "text": "Radar range depends on [1] ..."}   (many)
        {"type": "error",   "message": "..."}                          (only on failure)
        {"type": "done"}
    """
    with db.connect() as conn:
        api_key = documents.get_api_key(conn, user_id)
        has_docs = conn.execute("select exists (select 1 from tt_chunks where user_id = %s)", [user_id]).fetchone()[0]
    if not api_key and not settings.FAKE_LLM:
        raise HTTPException(400, "Add your Gemini API key in Settings first.")
    if not has_docs:
        raise HTTPException(404, "Nothing is indexed yet. Add PDFs and wait for them to be indexed.")
    history = trim_history(history_messages)

    def stream_events():
        try:
            llm = make_llm(api_key)
            embedder = make_embedder(api_key)
            with db.connect() as conn:       # released before the (slow) answer streams
                search_query, docs = retrieve(llm, conn, user_id, embedder, question, course, history)
            context, sources = number_sources(docs)
            yield _event("sources", sources=sources, search_query=search_query)

            # Memory, step 2: the model sees the recent conversation too.
            for chunk in llm.stream(build_messages(question, context, history)):
                text = message_text(chunk)
                if text:
                    yield _event("token", text=text)
        except Exception as e:
            yield _event("error", message=_friendly(e))
        yield _event("done")

    return StreamingResponse(stream_events(), media_type="application/x-ndjson")


# ===================================================================================== routes

@app.get("/api/health")
def health():
    return {"ok": True, "auth_mode": auth.AUTH_MODE, "fake_llm": settings.FAKE_LLM, "db": _startup["db"]}


@app.get("/api/me")
def me(user_id: str = Depends(current_user)):
    with db.connect() as conn:
        has_key = bool(documents.get_api_key(conn, user_id))
        docs = documents.list_documents(conn, user_id)
    status = documents.overall_status(docs)
    # Resume unfinished indexing (after a restart, a sleep, or the next day's quota).
    counts = status.get("counts", {})
    if (has_key or settings.EMBEDDINGS == "fake") and (counts.get("pending") or counts.get("processing")):
        indexer.ensure_worker(user_id)        # no-op if a worker is already running
    return {
        "user_id": user_id,
        "status": status["status"],
        "message": status["message"],
        "indexing": indexer.is_running(user_id),
        "has_api_key": has_key or settings.FAKE_LLM,
        "courses": documents.courses_with_files(docs),
        "max_upload_mb": settings.MAX_UPLOAD_MB,
    }


@app.put("/api/me/key")
def set_key(payload: KeyPayload, user_id: str = Depends(current_user)):
    key = payload.gemini_api_key.strip()
    if not key:
        raise HTTPException(400, "The API key is empty.")
    if len(key) > 200 or any(c.isspace() for c in key):
        raise HTTPException(400, "That doesn't look like a Gemini API key.")
    with db.connect() as conn:
        documents.save_api_key(conn, user_id, key)
    indexer.ensure_worker(user_id, force=True)       # a new key may lift a quota pause
    return {"saved": True}


@app.post("/api/me/upload")
async def upload_pdfs(course: str = Form(...), files: List[UploadFile] = File(...),
                      user_id: str = Depends(current_user)):
    """Store PDFs in a course and queue them for indexing (starts automatically).

    A file name may include sub-folders ("Chapter 1/slides.pdf"), e.g. from a folder upload.
    """
    try:
        course = documents.clean_course(course)
    except ValueError as e:
        raise HTTPException(400, str(e))
    saved = []
    for upload in files:
        try:
            name = documents.clean_name(upload.filename or "document.pdf")
        except ValueError as e:
            raise HTTPException(400, str(e))
        data = await upload.read()
        if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"{upload.filename} is larger than {settings.MAX_UPLOAD_MB} MB.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, f"{upload.filename} is not a valid PDF.")
        with db.connect() as conn:
            saved.append(documents.add_document(conn, user_id, course, name, data)["source"])
        del data
    indexer.ensure_worker(user_id)
    return {"saved": saved}


@app.delete("/api/me/files")
def delete_pdf(source: str = Query(...), user_id: str = Depends(current_user)):
    try:
        with db.connect() as conn:
            deleted = documents.delete_document(conn, user_id, source)
    except ValueError:
        deleted = False
    if not deleted:
        raise HTTPException(404, "File not found.")
    return {"deleted": source}


@app.post("/api/me/index")
def index_pdfs(source: Optional[str] = Query(None), user_id: str = Depends(current_user)):
    """Index again: one PDF (?source=...) or all of them. Also resumes a paused queue."""
    try:
        with db.connect() as conn:
            queued = documents.reindex(conn, user_id, source)
    except ValueError:
        raise HTTPException(404, "File not found.")
    if source and not queued:
        raise HTTPException(409, "That PDF is being indexed right now.")
    indexer.ensure_worker(user_id, force=True)
    return {"queued": queued}


@app.get("/api/me/page")
def page_image(source: str = Query(...), page: int = Query(..., ge=1), highlight: Optional[str] = Query(None),
               user_id: str = Depends(current_user)):
    """PNG of one page of the user's PDF, with the cited passage highlighted."""
    try:
        with db.connect() as conn:
            doc = documents.find_document(conn, user_id, source)
    except ValueError:
        doc = None
    if not doc or not doc["storage_path"]:
        raise HTTPException(404, "File not found.")
    try:
        png, pages = render_page(get_storage().get(doc["storage_path"]), page, highlight)
    except IndexError as e:
        raise HTTPException(404, str(e))
    return Response(png, media_type="image/png", headers={
        "Cache-Control": "private, max-age=3600",
        "X-Page-Count": str(pages),
    })


@app.post("/api/chat")
def chat(payload: ChatPayload, user_id: str = Depends(current_user)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Ask a question.")
    return answer_stream(user_id, question[:2000], payload.course, [m.model_dump() for m in payload.history])


@app.get("/")
def root():
    return {"name": "Teleco-Tutor API", "time": int(time.time())}
