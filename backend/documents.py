"""The user's documents and settings, in Postgres (metadata) + storage (PDF files).

A document is addressed by its "source": "<course>/<name>", e.g.
"Radar/Course/chap1.pdf" (name may contain sub-folders). Files are stored under
"<user_id>/<document_id>.pdf", so odd file names never reach the storage API.
"""
import re
from typing import List, Optional

import secretbox
from storage import get_storage

STATUS_PENDING, STATUS_PROCESSING, STATUS_READY, STATUS_ERROR = "pending", "processing", "ready", "error"

_BAD_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')


def clean_part(name: str) -> str:
    """One path segment: no characters that are invalid on Windows/Linux, no '..'."""
    cleaned = _BAD_CHARS.sub("_", name.replace("/", "_")).strip(" .")
    return cleaned or "_"


def clean_course(course: str) -> str:
    course = (course or "").strip()
    if not course:
        raise ValueError("Choose a course name.")
    return clean_part(course)[:100]


def clean_name(name: str) -> str:
    """A file name, possibly with sub-folders ("Course/Chapter 1/chap1.pdf")."""
    parts = [clean_part(p) for p in re.split(r"[\\/]+", name or "") if p.strip(" .")]
    if not parts:
        raise ValueError("Missing file name.")
    path = "/".join(parts)[:300]
    if not path.lower().endswith(".pdf"):
        raise ValueError(f"{name}: only PDF files can be uploaded.")
    return path


def split_source(source: str):
    course, _, name = (source or "").partition("/")
    if not course or not name:
        raise ValueError("Invalid source.")
    return course, name


# ----------------------------------------------------------------------------- profile

def get_api_key(conn, user_id) -> Optional[str]:
    row = conn.execute("select gemini_key_enc from tt_profiles where user_id = %s", [user_id]).fetchone()
    return secretbox.decrypt(row[0]) if row and row[0] else None


def save_api_key(conn, user_id, key: str) -> None:
    conn.execute(
        "insert into tt_profiles (user_id, gemini_key_enc) values (%s, %s) "
        "on conflict (user_id) do update set gemini_key_enc = excluded.gemini_key_enc, updated_at = now()",
        [user_id, secretbox.encrypt(key)])


# ----------------------------------------------------------------------------- documents

def add_document(conn, user_id, course: str, name: str, data: bytes) -> dict:
    """Store a PDF and queue it for indexing. Re-uploading the same name replaces it."""
    row = conn.execute(
        """insert into tt_documents (user_id, course, name, storage_path, size_bytes, status, message)
           values (%s, %s, %s, '', %s, 'pending', 'Waiting to be indexed')
           on conflict (user_id, course, name) do update
             set size_bytes = excluded.size_bytes, status = 'pending', message = 'Waiting to be indexed',
                 pages = null, chunk_count = 0, ocr_pages = 0, updated_at = now()
           returning id""", [user_id, course, name, len(data)]).fetchone()
    doc_id = row[0]
    key = f"{user_id}/{doc_id}.pdf"
    get_storage().put(key, data)
    conn.execute("delete from tt_chunks where document_id = %s", [doc_id])
    conn.execute("update tt_documents set storage_path = %s where id = %s", [key, doc_id])
    return {"id": str(doc_id), "source": f"{course}/{name}"}


def find_document(conn, user_id, source: str) -> Optional[dict]:
    course, name = split_source(source)
    row = conn.execute(
        "select id, storage_path, pages from tt_documents where user_id = %s and course = %s and name = %s",
        [user_id, course, name]).fetchone()
    return {"id": row[0], "storage_path": row[1], "pages": row[2]} if row else None


def delete_document(conn, user_id, source: str) -> bool:
    doc = find_document(conn, user_id, source)
    if not doc:
        return False
    conn.execute("delete from tt_documents where id = %s", [doc["id"]])     # chunks cascade
    if doc["storage_path"]:
        get_storage().delete(doc["storage_path"])
    return True


def reindex(conn, user_id, source: Optional[str] = None) -> int:
    """Queue one document (or all of them) to be indexed again."""
    sql = ("update tt_documents set status = 'pending', message = 'Waiting to be indexed', updated_at = now() "
           "where user_id = %s and status <> 'processing'")
    params = [user_id]
    if source:
        course, name = split_source(source)
        sql += " and course = %s and name = %s"
        params += [course, name]
    return conn.execute(sql, params).rowcount


def list_documents(conn, user_id) -> List[dict]:
    rows = conn.execute(
        """select course, name, status, message, pages, chunk_count, ocr_pages, size_bytes, updated_at
           from tt_documents where user_id = %s order by course, name""", [user_id]).fetchall()
    return [{
        "course": r[0], "name": r[1], "source": f"{r[0]}/{r[1]}", "file_name": r[1].rsplit("/", 1)[-1],
        "status": r[2], "message": r[3], "pages": r[4], "chunks": r[5], "ocr_pages": r[6],
        "size_bytes": r[7], "updated_at": r[8].isoformat() if r[8] else None,
    } for r in rows]


def courses_with_files(docs: List[dict]) -> List[dict]:
    """Group documents by course: [{"name": "Radar", "files": [...]}, ...]."""
    courses = {}
    for d in docs:
        courses.setdefault(d["course"], []).append(d)
    return [{"name": c, "files": files} for c, files in sorted(courses.items())]


def overall_status(docs: List[dict]) -> dict:
    """One status for the whole library, for the header and onboarding."""
    if not docs:
        return {"status": "empty", "message": "No PDFs yet."}
    counts = {s: sum(d["status"] == s for d in docs) for s in
              (STATUS_PENDING, STATUS_PROCESSING, STATUS_READY, STATUS_ERROR)}
    busy = counts[STATUS_PENDING] + counts[STATUS_PROCESSING]
    if busy:
        current = next((d for d in docs if d["status"] == STATUS_PROCESSING), None)
        paused = next((d for d in docs if d["status"] == STATUS_PENDING and d["message"].startswith("Paused")), None)
        message = (current["message"] if current else paused["message"] if paused else "Waiting to be indexed")
        return {"status": "processing", "message": f"{counts[STATUS_READY]}/{len(docs)} indexed. {message}",
                "counts": counts}
    if counts[STATUS_READY]:
        note = f" {counts[STATUS_ERROR]} failed." if counts[STATUS_ERROR] else ""
        return {"status": "ready", "message": f"{counts[STATUS_READY]} PDFs indexed.{note}", "counts": counts}
    return {"status": "error", "message": "Indexing failed for every PDF.", "counts": counts}
