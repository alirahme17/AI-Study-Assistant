"""Embedding wrapper (prefixes, lazy loading) and per-database model tracking."""
import sys
import types

from fastapi.testclient import TestClient

import embeddings
import main
from workspace import STATUS_READY, embedding_model_for, save_user_api_key, write_status


class FakeSentenceTransformer:
    loaded = []

    def __init__(self, name):
        FakeSentenceTransformer.loaded.append(name)
        self.seen = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        import numpy as np
        assert normalize_embeddings is True
        self.seen.extend(texts)
        return [np.array([float(len(t)), 1.0]) for t in texts]


def install_fake(monkeypatch):
    FakeSentenceTransformer.loaded = []
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))


def test_e5_models_get_query_and_passage_prefixes(monkeypatch):
    install_fake(monkeypatch)
    emb = embeddings.SentenceTransformerEmbeddings("intfloat/multilingual-e5-small")
    emb.embed_documents(["La bande passante"])
    emb.embed_query("antenna bandwidth")
    assert emb.model.seen == ["passage: La bande passante", "query: antenna bandwidth"]


def test_minilm_gets_no_prefixes(monkeypatch):
    install_fake(monkeypatch)
    emb = embeddings.SentenceTransformerEmbeddings(embeddings.LEGACY_MODEL)
    assert emb.embed_query("radar") == [5.0, 1.0]
    assert emb.model.seen == ["radar"]


def test_model_loads_lazily(monkeypatch):
    install_fake(monkeypatch)
    emb = embeddings.SentenceTransformerEmbeddings("intfloat/multilingual-e5-small")
    assert FakeSentenceTransformer.loaded == []          # opening a DB must not load the model
    emb.embed_query("x")
    emb.embed_query("y")
    assert FakeSentenceTransformer.loaded == ["intfloat/multilingual-e5-small"]   # loaded once


def test_default_for_new_indexes_is_multilingual():
    assert embeddings.DEFAULT_MODEL == "intfloat/multilingual-e5-small"


def test_old_databases_keep_using_minilm():
    write_status("alice", STATUS_READY, db_dir="chroma_db")           # built before this change
    assert embedding_model_for("alice") == embeddings.LEGACY_MODEL


def test_model_is_recorded_and_kept_across_status_updates():
    write_status("bob", STATUS_READY, db_dir="chroma_db_new", embedding_model="intfloat/multilingual-e5-small")
    write_status("bob", "processing", "re-indexing...")                 # later status change
    assert embedding_model_for("bob") == "intfloat/multilingual-e5-small"


def test_query_opens_database_with_its_own_model(isolated_workspaces, monkeypatch):
    save_user_api_key("carol", "KEY")
    (isolated_workspaces / "workspaces" / "carol" / "chroma_db_x").mkdir(parents=True)
    write_status("carol", STATUS_READY, db_dir="chroma_db_x", embedding_model="intfloat/multilingual-e5-small")
    opened = []

    class Store:
        def similarity_search(self, query, k, filter=None):
            return []

    def fake_open(path, model=embeddings.LEGACY_MODEL):
        opened.append(model)
        return Store()

    class LLM:
        def stream(self, messages):
            return iter([])
    monkeypatch.setattr(main, "open_vector_store", fake_open)
    monkeypatch.setattr(main, "make_llm", lambda key, model=None: LLM())
    monkeypatch.setattr(main, "safe_keyword_index", lambda path: None)
    TestClient(main.app).post("/api/query", json={"username": "carol", "question": "q"})
    assert opened == ["intfloat/multilingual-e5-small"]


def test_pipeline_records_model_of_new_build(isolated_workspaces, monkeypatch):
    import time
    (isolated_workspaces / "workspaces" / "dan" / "data").mkdir(parents=True)
    used = []

    def fake_build(username, progress=None, embedding_model=None):
        used.append(embedding_model)
        (isolated_workspaces / "workspaces" / username / "chroma_db_e5").mkdir()
        return "chroma_db_e5"
    monkeypatch.setattr(main, "build_user_database", fake_build)
    client = TestClient(main.app)
    client.post("/api/rebuild/dan")
    for _ in range(200):
        if client.get("/api/status/dan").json()["status"] != "processing":
            break
        time.sleep(0.01)
    assert used == [main.EMBEDDING_MODEL]
    assert embedding_model_for("dan") == main.EMBEDDING_MODEL
