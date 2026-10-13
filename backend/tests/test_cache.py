"""Quota is never spent twice: embeddings and OCR text are cached across pauses and re-indexing."""
import pymupdf
import pytest

import auth
import db
import documents
import indexer
from cache import CachingEmbedder
from conftest import wait_for_indexing
from providers import FakeEmbedder
from quota import QuotaExhausted

USER = auth.dev_user_id("cachetest")


class CountingEmbedder(FakeEmbedder):
    """Fake embedder that counts texts sent to the 'API' and can run out of quota."""

    def __init__(self, quota=None):
        super().__init__()
        self.sent, self.quota = 0, quota

    def embed_documents(self, texts, title="none"):
        if self.quota is not None and self.sent + len(texts) > self.quota:
            raise QuotaExhausted("429 RESOURCE_EXHAUSTED PerDay")
        self.sent += len(texts)
        return super().embed_documents(texts, title)

    def embed_query(self, text):
        self.sent += 1
        return super().embed_query(text)


class CountingOCR:
    model = "fake-ocr"

    def __init__(self, quota=None):
        self.calls, self.quota = 0, quota

    def read_page(self, page):
        if self.quota is not None and self.calls >= self.quota:
            raise QuotaExhausted("PerDay")
        self.calls += 1
        return f"Handwritten notes on page {page.number + 1} about antenna gain."


def long_pdf(pages=12, lines=30):
    doc = pymupdf.open()
    for p in range(pages):
        text = "\n".join(f"Page {p} line {i}: the radar equation relates power, gain and range {p * 100 + i}."
                         for i in range(lines))
        doc.new_page().insert_textbox(pymupdf.Rect(36, 36, 560, 800), text, fontsize=7)
    return doc.tobytes()


def scanned_pdf(pages=4):
    doc = pymupdf.open()
    for _ in range(pages):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
        pix.clear_with(200)
        doc.new_page().insert_image(pymupdf.Rect(72, 72, 300, 300), pixmap=pix)
    return doc.tobytes()


def add(name, data):
    with db.connect() as conn:
        documents.add_document(conn, USER, "Radar", name, data)


def run(embedder, ocr=None):
    indexer.ensure_worker(USER, force=True, embedder=embedder, ocr=ocr)
    wait_for_indexing(USER)


def statuses():
    with db.connect() as conn:
        return {d["name"]: d for d in documents.list_documents(conn, USER)}


def test_paused_pdf_resumes_without_re_embedding_finished_batches(database):
    add("big.pdf", long_pdf(pages=40))
    first = CountingEmbedder(quota=60)              # enough for one batch of 50, not the whole PDF
    run(first)
    doc = statuses()["big.pdf"]
    assert doc["status"] == "pending" and doc["message"].startswith("Paused")
    with db.connect() as conn:
        saved = conn.execute("select count(*) from tt_embedding_cache").fetchone()[0]
    assert saved == 50                               # the finished batch was kept

    second = CountingEmbedder()                     # next day
    indexer._paused_until.clear()
    run(second)
    doc = statuses()["big.pdf"]
    assert doc["status"] == "ready"
    assert second.sent == doc["chunks"] - 50         # only what was never embedded


def test_reindexing_the_same_pdf_costs_nothing(database):
    add("a.pdf", long_pdf(pages=3))
    run(CountingEmbedder())
    again = CountingEmbedder()
    with db.connect() as conn:
        documents.reindex(conn, USER)
    run(again)
    assert statuses()["a.pdf"]["status"] == "ready" and again.sent == 0


def test_scanned_pages_are_read_once_even_across_quota_pauses_and_reuploads(database):
    data = scanned_pdf(pages=4)
    add("scan.pdf", data)
    ocr = CountingOCR(quota=2)
    run(CountingEmbedder(), ocr)
    doc = statuses()["scan.pdf"]
    assert ocr.calls == 2 and "2 scanned page(s) not read" in doc["message"]

    ocr2 = CountingOCR()                            # quota reset: retry the file
    with db.connect() as conn:
        documents.reindex(conn, USER, "Radar/scan.pdf")
    run(CountingEmbedder(), ocr2)
    doc = statuses()["scan.pdf"]
    assert ocr2.calls == 2                           # only the 2 pages never read
    assert doc["ocr_pages"] == 4 and "not read" not in doc["message"]

    ocr3 = CountingOCR()
    add("scan-copy.pdf", data)                      # same file uploaded again under another name
    run(CountingEmbedder(), ocr3)
    assert ocr3.calls == 0 and statuses()["scan-copy.pdf"]["ocr_pages"] == 4


def test_query_embeddings_are_cached_per_user(database):
    inner = CountingEmbedder()
    emb = CachingEmbedder(inner, USER)
    a = emb.embed_query("What is the radar equation?")
    b = emb.embed_query("What is the radar equation?")
    other = CachingEmbedder(inner, auth.dev_user_id("someone-else"))
    other.embed_query("What is the radar equation?")
    assert a == pytest.approx(b, abs=1e-6) and inner.sent == 2   # once for each user (stored as float4)


def test_quota_error_during_query_is_not_cached(database):
    class Broken(CountingEmbedder):
        def embed_query(self, text):
            raise QuotaExhausted("PerDay")
    with pytest.raises(QuotaExhausted):
        CachingEmbedder(Broken(), USER).embed_query("q")
    with db.connect() as conn:
        assert conn.execute("select count(*) from tt_embedding_cache").fetchone()[0] == 0
