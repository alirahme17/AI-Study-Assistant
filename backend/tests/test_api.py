"""The API used by the Next.js frontend, against a real Postgres (see conftest.py)."""
import json
import time
import uuid

import jwt
import pymupdf
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

import auth
import db
import indexer
import main
from conftest import wait_for_indexing
from quota import QuotaExhausted

ALICE = {"X-Dev-User": "alice"}
BOB = {"X-Dev-User": "bob"}
SECRET = "s3cret-for-tests-only-0123456789abcdef"


@pytest.fixture
def client(monkeypatch, database):
    monkeypatch.setattr(auth, "AUTH_MODE", "dev")
    monkeypatch.setattr(auth, "SUPABASE_URL", "")
    return TestClient(main.app)


def pdf_bytes(*pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


def upload(client, headers, course, files):
    """files: {name: bytes}"""
    return client.post("/api/me/upload", headers=headers, data={"course": course},
                       files=[("files", (name, data, "application/pdf")) for name, data in files.items()])


def files_of(body):
    return {f["source"]: f for c in body["courses"] for f in c["files"]}


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


class RecordingLLM:
    def __init__(self, rewrite="What is the maximum range of a radar?", tokens=("Range is ", "R_max [1].")):
        self.rewrite, self.tokens, self.invoked, self.streamed = rewrite, tokens, [], []

    def invoke(self, messages):
        self.invoked.append(messages)
        return AIMessage(self.rewrite)

    def stream(self, messages):
        self.streamed.append(messages)
        for t in self.tokens:
            yield AIMessage(t)


# ------------------------------------------------------------------------- auth ---

def test_requests_without_identity_are_rejected(client):
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/chat", json={"question": "q"}).status_code == 401


def test_supabase_hs256_token_gives_the_user_id(monkeypatch, database):
    monkeypatch.setattr(auth, "AUTH_MODE", "supabase")
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", SECRET)
    uid = "8f14e45f-ceea-467a-9575-0d1f1c2b3a4b"
    token = jwt.encode({"sub": uid, "aud": "authenticated", "exp": time.time() + 60}, SECRET, algorithm="HS256")
    r = TestClient(main.app).get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["user_id"] == uid


def test_supabase_rejects_bad_expired_non_uuid_and_dev_headers(monkeypatch, database):
    monkeypatch.setattr(auth, "AUTH_MODE", "supabase")
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", SECRET)
    c = TestClient(main.app)
    forged = jwt.encode({"sub": str(uuid.uuid4()), "aud": "authenticated"}, "wrong-secret-0123456789abcdefghij",
                        algorithm="HS256")
    expired = jwt.encode({"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": time.time() - 10}, SECRET,
                         algorithm="HS256")
    not_uuid = jwt.encode({"sub": "../etc", "aud": "authenticated"}, SECRET, algorithm="HS256")
    assert c.get("/api/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    r = c.get("/api/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401 and "expired" in r.json()["detail"]
    assert c.get("/api/me", headers={"Authorization": f"Bearer {not_uuid}"}).status_code == 401
    assert c.get("/api/me", headers=ALICE).status_code == 401          # dev header ignored in production


def test_supabase_asymmetric_token_via_jwks(monkeypatch, database):
    from cryptography.hazmat.primitives.asymmetric import ec
    private = ec.generate_private_key(ec.SECP256R1())
    monkeypatch.setattr(auth, "AUTH_MODE", "supabase")

    class FakeJWKS:
        def get_signing_key_from_jwt(self, token):
            return type("K", (), {"key": private.public_key()})()
    monkeypatch.setattr(auth, "_jwks_client", lambda: FakeJWKS())
    uid = str(uuid.uuid4())
    token = jwt.encode({"sub": uid, "aud": "authenticated"}, private, algorithm="ES256")
    assert TestClient(main.app).get("/api/me", headers={"Authorization": f"Bearer {token}"}).json()["user_id"] == uid


def test_dev_mode_refuses_to_run_with_supabase_configured(monkeypatch, database):
    monkeypatch.setattr(auth, "AUTH_MODE", "dev")
    monkeypatch.setattr(auth, "SUPABASE_URL", "https://example.supabase.co")
    assert TestClient(main.app).get("/api/me", headers=ALICE).status_code == 500


def test_health_reports_database(client):
    with client:                                   # runs the startup (schema + storage check)
        body = client.get("/api/health").json()
    assert body["ok"] and body["db"] == "ok"


# ---------------------------------------------------------------- key + library ---

def test_new_user(client):
    body = client.get("/api/me", headers=ALICE).json()
    assert body["status"] == "empty" and body["courses"] == [] and body["has_api_key"] is False


def test_key_is_stored_encrypted(client):
    assert client.put("/api/me/key", headers=ALICE, json={"gemini_api_key": "  AIza-test-key  "}).json() == {"saved": True}
    assert client.get("/api/me", headers=ALICE).json()["has_api_key"] is True
    assert client.get("/api/me", headers=BOB).json()["has_api_key"] is False
    with db.connect() as conn:
        stored = conn.execute("select gemini_key_enc from tt_profiles").fetchone()[0]
        assert "AIza" not in stored
        assert __import__("documents").get_api_key(conn, auth.dev_user_id("alice")) == "AIza-test-key"
    assert client.put("/api/me/key", headers=ALICE, json={"gemini_api_key": " "}).status_code == 400


def test_upload_indexes_automatically_and_lists_status(client):
    r = upload(client, ALICE, "Radar", {
        "Course/chap1.pdf": pdf_bytes("Radar basics: a pulse is transmitted.", "The radar equation gives range."),
        "notes.pdf": pdf_bytes("Doppler shift measures velocity."),
    })
    assert r.status_code == 200
    assert sorted(r.json()["saved"]) == ["Radar/Course/chap1.pdf", "Radar/notes.pdf"]
    wait_for_indexing()
    body = client.get("/api/me", headers=ALICE).json()
    files = files_of(body)
    assert body["status"] == "ready"
    chap1 = files["Radar/Course/chap1.pdf"]
    assert chap1["status"] == "ready" and chap1["pages"] == 2 and chap1["chunks"] == 2
    assert chap1["file_name"] == "chap1.pdf" and "2 pages" in chap1["message"]


def test_upload_rejects_non_pdfs_and_bad_names(client):
    r = client.post("/api/me/upload", headers=ALICE, data={"course": "Radar"},
                    files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400
    r = client.post("/api/me/upload", headers=ALICE, data={"course": "Radar"},
                    files=[("files", ("fake.pdf", b"not a pdf", "application/pdf"))])
    assert r.status_code == 400
    assert upload(client, ALICE, "  ", {"a.pdf": pdf_bytes("x")}).status_code == 400
    # Path tricks are neutralised into plain names inside the course.
    r = upload(client, ALICE, "../..", {"../../evil.pdf": pdf_bytes("x")})
    assert r.status_code == 200 and r.json()["saved"] == ["_/evil.pdf"]


def test_reupload_replaces_and_delete_removes_everything(client):
    upload(client, ALICE, "Radar", {"c.pdf": pdf_bytes("old content about sonar")})
    wait_for_indexing()
    upload(client, ALICE, "Radar", {"c.pdf": pdf_bytes("new content about lidar", "page two")})
    wait_for_indexing()
    with db.connect() as conn:
        contents = [r[0] for r in conn.execute("select content from tt_chunks order by page")]
    assert len(contents) == 2 and "lidar" in contents[0]
    assert client.delete("/api/me/files", headers=ALICE, params={"source": "Radar/c.pdf"}).status_code == 200
    with db.connect() as conn:
        assert conn.execute("select count(*) from tt_chunks").fetchone()[0] == 0
    assert client.get("/api/me", headers=ALICE).json()["courses"] == []
    assert client.delete("/api/me/files", headers=ALICE, params={"source": "Radar/c.pdf"}).status_code == 404


def test_users_cannot_see_each_others_files(client):
    upload(client, ALICE, "Radar", {"secret.pdf": pdf_bytes("alice only")})
    wait_for_indexing()
    assert client.get("/api/me", headers=BOB).json()["courses"] == []
    assert client.get("/api/me/page", headers=BOB, params={"source": "Radar/secret.pdf", "page": 1}).status_code == 404
    assert client.delete("/api/me/files", headers=BOB, params={"source": "Radar/secret.pdf"}).status_code == 404
    assert client.get("/api/me/page", headers=ALICE, params={"source": "Radar/secret.pdf", "page": 1}).status_code == 200


def test_page_image_with_highlight(client):
    upload(client, ALICE, "Radar", {"c1.pdf": pdf_bytes("Page one", "The radar range equation is here")})
    wait_for_indexing()
    r = client.get("/api/me/page", headers=ALICE,
                   params={"source": "Radar/c1.pdf", "page": 2, "highlight": "The radar range equation is here"})
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    assert r.headers["x-page-count"] == "2"
    assert client.get("/api/me/page", headers=ALICE, params={"source": "Radar/c1.pdf", "page": 9}).status_code == 404
    assert client.get("/api/me/page", headers=ALICE, params={"source": "nope", "page": 1}).status_code == 404


def test_reindex_one_file(client):
    upload(client, ALICE, "Radar", {"a.pdf": pdf_bytes("alpha"), "b.pdf": pdf_bytes("beta")})
    wait_for_indexing()
    assert client.post("/api/me/index", headers=ALICE, params={"source": "Radar/a.pdf"}).json() == {"queued": 1}
    wait_for_indexing()
    assert {f["status"] for f in files_of(client.get("/api/me", headers=ALICE).json()).values()} == {"ready"}
    assert client.post("/api/me/index", headers=ALICE, params={"source": "bad"}).status_code == 404


# ---------------------------------------------------------- resumable indexing ---

class QuotaAfter:
    """Embedder that runs out of daily quota after n documents."""

    def __init__(self, n):
        from providers import FakeEmbedder
        self.inner, self.left, self.model = FakeEmbedder(), n, "fake-hash-768"

    def embed_documents(self, texts, title="none"):
        if self.left == 0:
            raise QuotaExhausted("429 RESOURCE_EXHAUSTED PerDay")
        self.left -= 1
        return self.inner.embed_documents(texts)


def test_daily_quota_pauses_and_the_rest_resumes_later(client, monkeypatch):
    real_ensure = indexer.ensure_worker
    monkeypatch.setattr(indexer, "ensure_worker", lambda uid, force=False: real_ensure(uid, force, embedder=QuotaAfter(1)))
    upload(client, ALICE, "Radar", {"a.pdf": pdf_bytes("alpha"), "b.pdf": pdf_bytes("beta"), "c.pdf": pdf_bytes("gamma")})
    wait_for_indexing()
    body = client.get("/api/me", headers=ALICE).json()
    statuses = sorted(f["status"] for f in files_of(body).values())
    assert statuses == ["pending", "pending", "ready"]
    assert body["status"] == "processing" and "daily limit" in body["message"]
    assert indexer.paused_until(auth.dev_user_id("alice"))
    # Polling /api/me during the pause doesn't hammer the API...
    assert body["indexing"] is False
    # ...and the next day (pause over) everything finishes.
    monkeypatch.setattr(indexer, "ensure_worker", real_ensure)
    indexer._paused_until.clear()
    client.get("/api/me", headers=ALICE)
    wait_for_indexing()
    assert {f["status"] for f in files_of(client.get("/api/me", headers=ALICE).json()).values()} == {"ready"}


def test_documents_stuck_in_processing_after_a_restart_are_resumed(client):
    upload(client, ALICE, "Radar", {"a.pdf": pdf_bytes("alpha")})
    wait_for_indexing()
    with db.connect() as conn:          # simulate a crash mid-indexing
        conn.execute("update tt_documents set status = 'processing'")
    client.get("/api/me", headers=ALICE)
    wait_for_indexing()
    assert files_of(client.get("/api/me", headers=ALICE).json())["Radar/a.pdf"]["status"] == "ready"


def test_a_broken_pdf_fails_alone(client, monkeypatch):
    import storage
    upload(client, ALICE, "Radar", {"good.pdf": pdf_bytes("fine"), "bad.pdf": pdf_bytes("will break")})
    wait_for_indexing()
    with db.connect() as conn:
        key = conn.execute("select storage_path from tt_documents where name = 'bad.pdf'").fetchone()[0]
    storage.get_storage().put(key, b"%PDF-1.4 truncated garbage")
    client.post("/api/me/index", headers=ALICE)
    wait_for_indexing()
    files = files_of(client.get("/api/me", headers=ALICE).json())
    assert files["Radar/good.pdf"]["status"] == "ready"
    assert files["Radar/bad.pdf"]["status"] == "error" and files["Radar/bad.pdf"]["message"]


# --------------------------------------------------------------------------- chat ---

def test_chat_without_key_or_documents_explains_what_to_do(client):
    r = client.post("/api/chat", headers=ALICE, json={"question": "q"})
    assert r.status_code == 400 and "API key" in r.json()["detail"]
    client.put("/api/me/key", headers=ALICE, json={"gemini_api_key": "KEY"})
    r = client.post("/api/chat", headers=ALICE, json={"question": "q"})
    assert r.status_code == 404 and "indexed" in r.json()["detail"]


def _ready_library(client, monkeypatch, llm):
    client.put("/api/me/key", headers=ALICE, json={"gemini_api_key": "KEY"})
    upload(client, ALICE, "Radar", {"Course/chap1.pdf": pdf_bytes("Intro", "R_max formula: maximum range of a radar")})
    upload(client, ALICE, "Antenna", {"gain.pdf": pdf_bytes("Antenna gain and maximum directivity")})
    wait_for_indexing()
    monkeypatch.setattr(main, "make_llm", lambda key: llm)


def test_chat_streams_sources_then_tokens(client, monkeypatch):
    llm = RecordingLLM()
    _ready_library(client, monkeypatch, llm)
    r = client.post("/api/chat", headers=ALICE, json={"question": "What is the maximum range?", "course": "Radar"})
    evs = events(r)
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert [e["type"] for e in evs] == ["sources", "token", "token", "done"]
    first = evs[0]["sources"][0]
    assert first["source"] == "Radar/Course/chap1.pdf" and first["page"] == 2 and first["file_name"] == "chap1.pdf"
    assert {s["course"] for s in evs[0]["sources"]} == {"Radar"}               # course filter
    assert "".join(e["text"] for e in evs if e["type"] == "token") == "Range is R_max [1]."


def test_follow_up_uses_history_for_search_and_answer(client, monkeypatch):
    llm = RecordingLLM()
    _ready_library(client, monkeypatch, llm)
    evs = events(client.post("/api/chat", headers=ALICE, json={
        "question": "And its range?",
        "history": [{"role": "user", "content": "What is a radar?"}, {"role": "assistant", "content": "A sensor [1]."}]}))
    assert evs[0]["search_query"] == "What is the maximum range of a radar?"      # rewritten for search
    answer_prompt = llm.streamed[0]
    assert any(isinstance(m, HumanMessage) and m.content == "What is a radar?" for m in answer_prompt)
    assert "And its range?" in answer_prompt[-1].content                          # original wording answered


def test_llm_errors_are_reported_as_events(client, monkeypatch):
    class Failing(RecordingLLM):
        def stream(self, messages):
            raise QuotaExhausted("PerDay")
            yield
    _ready_library(client, monkeypatch, Failing())
    evs = events(client.post("/api/chat", headers=ALICE, json={"question": "range?"}))
    assert evs[-2]["type"] == "error" and "daily limit" in evs[-2]["message"]
    assert evs[-1]["type"] == "done"


def test_chat_end_to_end_with_offline_demo_model(client, monkeypatch):
    monkeypatch.setattr(main.settings, "FAKE_LLM", True)
    upload(client, ALICE, "Radar", {"c.pdf": pdf_bytes("The maximum unambiguous range depends on the PRF of the radar.")})
    wait_for_indexing()
    evs = events(client.post("/api/chat", headers=ALICE, json={"question": "What limits the unambiguous range?"}))
    answer = "".join(e["text"] for e in evs if e["type"] == "token")
    assert evs[0]["sources"][0]["source"] == "Radar/c.pdf" and "[1]" in answer
