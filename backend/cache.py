"""Never pay twice for the same Gemini call: cached embeddings and OCR text, in Postgres.

The free Gemini tier has small daily limits, and indexing can be interrupted (daily
limit, server sleep, re-upload, "Retry"). Without a cache, an interrupted PDF starts
from zero the next day: every scanned page is read again and every passage embedded
again. With it:

- embeddings are saved after every batch, keyed by a hash of (model, dims, exact input),
  so a resumed or re-indexed PDF only embeds passages that were never embedded;
- OCR text is saved per page, keyed by the PDF's content hash, so re-uploading or
  re-indexing the same file never reads its scanned pages again;
- repeated questions (e.g. the evaluation set) reuse their query embedding.

Entries are per user: each user's cache was paid for with their own key.
"""
import hashlib
import json
from typing import Dict, List

import db
from providers import EMBED_BATCH


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pdf_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CachingEmbedder:
    """Wraps an embedder (GeminiEmbedder / FakeEmbedder) with a per-user Postgres cache."""

    def __init__(self, inner, user_id):
        self.inner, self.user_id = inner, user_id
        self.model = inner.model
        self.dims = getattr(inner, "dims", 768)
        self.api_calls = 0          # texts actually sent to the API (for tests/logs)

    def _key(self, kind: str, title: str, text: str) -> str:
        return _sha(f"{self.model}|{self.dims}|{kind}|{title}|{text}")

    def _load(self, keys: List[str]) -> Dict[str, List[float]]:
        if not keys:
            return {}
        with db.connect() as conn:
            rows = conn.execute(
                "select key, embedding::text from tt_embedding_cache where user_id = %s and key = any(%s)",
                [self.user_id, list(set(keys))]).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def _save(self, items: Dict[str, List[float]]) -> None:
        if not items:
            return
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into tt_embedding_cache (user_id, key, embedding) values (%s, %s, %s::extensions.vector) "
                    "on conflict do nothing",
                    [(self.user_id, k, db.vector_literal(v)) for k, v in items.items()])

    def embed_documents(self, texts: List[str], title: str = "none") -> List[List[float]]:
        keys = [self._key("doc", title, t) for t in texts]
        found = self._load(keys)
        missing = [(k, t) for k, t in zip(keys, texts) if k not in found]
        # unique texts only, embedded batch by batch, each batch saved at once: if the daily
        # quota runs out halfway, the finished batches are kept for next time.
        missing = list(dict(missing).items())
        for start in range(0, len(missing), EMBED_BATCH):
            batch = missing[start:start + EMBED_BATCH]
            vectors = self.inner.embed_documents([t for _, t in batch], title=title)
            self.api_calls += len(batch)
            new = {k: v for (k, _), v in zip(batch, vectors)}
            self._save(new)
            found.update(new)
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> List[float]:
        key = self._key("query", "", text)
        found = self._load([key])
        if key in found:
            return found[key]
        vector = self.inner.embed_query(text)
        self.api_calls += 1
        self._save({key: vector})
        return vector


class CachingOCR:
    """Wraps an OCR reader for one PDF: pages already read (even blank ones) are never re-read."""

    def __init__(self, inner, user_id, pdf_hash: str):
        self.inner, self.user_id, self.pdf_hash = inner, user_id, pdf_hash
        self.model = getattr(inner, "model", "ocr")
        self.api_calls = 0
        with db.connect() as conn:
            rows = conn.execute(
                "select page, text from tt_ocr_cache where user_id = %s and pdf_sha = %s and model = %s",
                [user_id, pdf_hash, self.model]).fetchall()
        self.pages = {p: t for p, t in rows}

    def read_page(self, page) -> str:
        number = page.number + 1
        if number in self.pages:
            return self.pages[number]
        text = self.inner.read_page(page)          # may raise QuotaExhausted: nothing cached then
        self.api_calls += 1
        self.pages[number] = text
        with db.connect() as conn:
            conn.execute(
                "insert into tt_ocr_cache (user_id, pdf_sha, model, page, text) values (%s, %s, %s, %s, %s) "
                "on conflict do nothing", [self.user_id, self.pdf_hash, self.model, number, text])
        return text
