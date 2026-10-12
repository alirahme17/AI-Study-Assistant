"""Gemini embeddings / OCR request shapes (with a fake client), fake embedder, quota handling."""
import math
from types import SimpleNamespace

import pytest

from providers import FakeEmbedder, GeminiEmbedder, GeminiOCR
from quota import QuotaExhausted, with_retry


class FakeModels:
    def __init__(self, fail=None):
        self.calls, self.fail = [], list(fail or [])

    def embed_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        if self.fail:
            raise self.fail.pop(0)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0] + [0.0] * (config.output_dimensionality - 2))
                                           for _ in contents])

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        return SimpleNamespace(text=" Page text \n")


def client(models):
    return SimpleNamespace(models=models)


def test_each_text_is_its_own_content_with_prefixes_batched_and_normalized():
    models = FakeModels()
    emb = GeminiEmbedder("key", client=client(models))
    vectors = emb.embed_documents([f"chunk {i}" for i in range(120)], title="chap1")
    assert len(vectors) == 120
    assert [len(c.contents) for c in models.calls] == [50, 50, 20]          # batched
    first = models.calls[0].contents[0]
    assert first.parts[0].text == "title: chap1 | text: chunk 0"            # one Content per text
    assert models.calls[0].config.output_dimensionality == 768
    assert models.calls[0].model == "gemini-embedding-2"
    assert math.isclose(sum(v * v for v in vectors[0]), 1.0)                # re-normalised
    emb.embed_query("what is DSDV?")
    assert models.calls[-1].contents[0].parts[0].text == "task: search result | query: what is DSDV?"


def test_embedding_retries_rate_limits_and_stops_on_daily_quota(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    models = FakeModels(fail=[RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 20s")])
    assert len(GeminiEmbedder("key", client=client(models)).embed_query("x")) == 768
    daily = FakeModels(fail=[RuntimeError("429 RESOURCE_EXHAUSTED quotaId: EmbedContentRequestsPerDayPerProject")])
    with pytest.raises(QuotaExhausted):
        GeminiEmbedder("key", client=client(daily)).embed_query("x")
    assert len(daily.calls) == 1                                              # not retried


def test_wrong_number_of_embeddings_is_an_error():
    class Aggregating(FakeModels):
        def embed_content(self, model, contents, config):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0] * 768)])
    with pytest.raises(RuntimeError, match="Expected 2"):
        GeminiEmbedder("key", client=client(Aggregating())).embed_documents(["a", "b"])


def test_ocr_sends_page_image_and_returns_text():
    import pymupdf
    doc = pymupdf.open()
    doc.new_page(width=200, height=200)
    models = FakeModels()
    assert GeminiOCR("key", client=client(models)).read_page(doc[0]) == "Page text"
    part = models.calls[0].contents[0]
    assert part.inline_data.mime_type == "image/png" and part.inline_data.data.startswith(b"\x89PNG")
    assert models.calls[0].model == "gemini-3.5-flash-lite"


def test_fake_embedder_is_deterministic_and_keyword_sensitive():
    e = FakeEmbedder()
    a, b, c = e.embed_query("radar pulse"), e.embed_query("radar pulse"), e.embed_query("antenna gain")
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
    assert a == b and len(a) == 768
    assert dot(a, e.embed_documents(["the radar sends a pulse"])[0]) > dot(c, e.embed_documents(["the radar sends a pulse"])[0])


def test_with_retry_gives_up_on_non_transient_errors():
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("bad request")
    with pytest.raises(ValueError):
        with_retry(boom, sleep=lambda s: None)
    assert len(calls) == 1
