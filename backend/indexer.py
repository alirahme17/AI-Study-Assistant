"""Resumable indexing queue: one row per PDF in tt_documents, processed one at a time.

    pending -> processing -> ready
                          -> error     (this PDF failed; the others carry on)
                          -> pending   (paused: Gemini's daily limit ran out)

Why a queue in the database instead of one long job: the free host sleeps after
15 minutes idle, restarts on deploys, and Gemini's free tier has daily limits.
With per-PDF state nothing is lost - whatever was indexed stays searchable, and
the rest continues the next time the user opens the app (GET /api/me calls
ensure_worker), after a restart, or once the quota resets.
"""
import threading
import time
import traceback
from typing import Callable, Optional

import db
import documents
from cache import CachingEmbedder, CachingOCR, pdf_sha
from process import chunk_pdf
from providers import make_embedder, make_ocr
from quota import QuotaExhausted
from storage import get_storage

PAUSE_SECONDS = 30 * 60      # after a daily-quota stop, don't retry for a while

_lock = threading.Lock()
_running: set = set()                 # user ids with a worker thread in this process
_paused_until: dict = {}              # user id -> time.time() before which we don't retry

PAUSED_MESSAGE = ("Paused: your Gemini API key reached its free daily limit{what}. "
                  "Indexing continues after the daily reset (midnight Pacific time, 09:00 in France) "
                  "- open the app then; nothing done so far is lost.")


def _paused_message(error) -> str:
    text = str(error)
    what = " for embeddings" if "mbed" in text else " for reading scanned pages" if "enerate" in text else ""
    return PAUSED_MESSAGE.format(what=what)


def claim_next(conn, user_id):
    """Atomically take the oldest pending PDF of this user (SKIP LOCKED: safe with several workers)."""
    return conn.execute(
        """update tt_documents set status = 'processing', message = 'Starting...', updated_at = now()
           where id = (select id from tt_documents where user_id = %s and status = 'pending'
                       order by created_at, id for update skip locked limit 1)
           returning id, course, name, storage_path""", [user_id]).fetchone()


def _set_message(doc_id, message: str) -> None:
    with db.connect() as conn:
        conn.execute("update tt_documents set message = %s, updated_at = now() "
                     "where id = %s and status = 'processing'", [message[:500], doc_id])


def index_document(doc, embedder, ocr, progress: Callable[[str], None]) -> dict:
    """Read, chunk, embed and store one PDF. Raises QuotaExhausted to pause."""
    doc_id, course, name, storage_path = doc
    data = get_storage().get(storage_path)
    if ocr is not None and getattr(embedder, "user_id", None) is not None:
        ocr = CachingOCR(ocr, embedder.user_id, pdf_sha(data))   # scanned pages are read only once, ever

    last = [0.0]

    def on_page(n, total):
        if time.time() - last[0] > 2:          # don't write to the database for every page
            last[0] = time.time()
            progress(f"Reading page {n}/{total} of {name}")

    pdf = chunk_pdf(data, ocr=ocr, on_page=on_page)
    del data
    progress(f"Embedding {len(pdf.chunks)} passages of {name}")
    title = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    vectors = embedder.embed_documents([c.content for c in pdf.chunks], title=title) if pdf.chunks else []

    notes = []
    if pdf.ocr_pages:
        notes.append(f"{pdf.ocr_pages} scanned page(s) read with OCR")
    if pdf.ocr_skipped:
        reason = ("OCR daily limit reached" if pdf.ocr_quota_hit
                  else "Gemini was unavailable" if pdf.ocr_errors else "")
        notes.append(f"{pdf.ocr_skipped} scanned page(s) not read"
                     + (f" ({reason} - use Retry later)" if reason and ocr is not None else ""))
    if not pdf.chunks:
        notes.append("no readable text found")
    message = f"{pdf.pages} pages, {len(pdf.chunks)} passages" + ("; " + "; ".join(notes) if notes else "")

    with db.connect() as conn:
        with conn.transaction():
            conn.execute("delete from tt_chunks where document_id = %s", [doc_id])
            with conn.cursor() as cur:
                cur.executemany(
                    """insert into tt_chunks (user_id, document_id, course, page, chunk_index, content, ocr, embedding)
                       select user_id, id, course, %s, %s, %s, %s, %s::extensions.vector from tt_documents where id = %s""",
                    [(c.page, c.chunk_index, c.content, c.ocr, db.vector_literal(v), doc_id)
                     for c, v in zip(pdf.chunks, vectors)])
            updated = conn.execute(
                """update tt_documents set status = 'ready', message = %s, pages = %s, chunk_count = %s,
                          ocr_pages = %s, embedding_model = %s, updated_at = now()
                   where id = %s and status = 'processing'""",
                [message, pdf.pages, len(pdf.chunks), pdf.ocr_pages, embedder.model, doc_id]).rowcount
            if not updated:
                # Replaced or deleted while we were working: discard this result.
                raise _Superseded()
    return {"chunks": len(pdf.chunks), "pages": pdf.pages}


class _Superseded(Exception):
    pass


def run_worker(user_id, embedder=None, ocr=None) -> None:
    """Index this user's pending PDFs until none are left (or the daily quota runs out)."""
    try:
        if embedder is None:
            with db.connect() as conn:
                api_key = documents.get_api_key(conn, user_id)
            embedder = make_embedder(api_key)
            ocr = make_ocr(api_key)
        if not isinstance(embedder, CachingEmbedder):
            embedder = CachingEmbedder(embedder, user_id)        # passages are embedded only once, ever
        while True:
            with db.connect() as conn:
                doc = claim_next(conn, user_id)
            if doc is None:
                return
            doc_id = doc[0]
            try:
                index_document(doc, embedder, ocr, lambda m: _set_message(doc_id, m))
            except QuotaExhausted as e:
                print(f"Indexing paused for {user_id}: daily quota reached ({e})")
                with db.connect() as conn:
                    conn.execute("update tt_documents set status = 'pending', message = %s, updated_at = now() "
                                 "where id = %s and status = 'processing'", [_paused_message(e), doc_id])
                with _lock:
                    _paused_until[user_id] = time.time() + PAUSE_SECONDS
                return
            except _Superseded:
                continue
            except Exception as e:
                traceback.print_exc()
                with db.connect() as conn:
                    conn.execute("update tt_documents set status = 'error', message = %s, updated_at = now() "
                                 "where id = %s and status = 'processing'",
                                 [f"{type(e).__name__}: {e}"[:500], doc_id])
    except Exception as e:
        # No key, database unreachable...: leave PDFs pending, report on the first one.
        traceback.print_exc()
        try:
            with db.connect() as conn:
                conn.execute("update tt_documents set message = %s where user_id = %s and status = 'pending'",
                             [f"Can't index yet: {e}"[:500], user_id])
        except Exception:
            pass
    finally:
        with _lock:
            _running.discard(user_id)


def ensure_worker(user_id, force: bool = False, **worker_kwargs) -> bool:
    """Start a background worker for this user if there is pending work and none is running.

    PDFs left in 'processing' by a server restart are put back in the queue first
    (this server is the only worker, so 'processing' without a thread here is stale).
    """
    with _lock:
        if user_id in _running:
            return False
        if not force and _paused_until.get(user_id, 0) > time.time():
            return False
        _paused_until.pop(user_id, None)
        _running.add(user_id)
    try:
        with db.connect() as conn:
            conn.execute("update tt_documents set status = 'pending', message = 'Waiting to be indexed (resumed)' "
                         "where user_id = %s and status = 'processing'", [user_id])
            has_work = conn.execute("select exists (select 1 from tt_documents where user_id = %s "
                                    "and status = 'pending')", [user_id]).fetchone()[0]
        if not has_work:
            with _lock:
                _running.discard(user_id)
            return False
        threading.Thread(target=run_worker, args=(user_id,), kwargs=worker_kwargs, daemon=True).start()
        return True
    except Exception:
        with _lock:
            _running.discard(user_id)
        raise


def is_running(user_id) -> bool:
    with _lock:
        return user_id in _running


def paused_until(user_id) -> Optional[float]:
    with _lock:
        return _paused_until.get(user_id)
