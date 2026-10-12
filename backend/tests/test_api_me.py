"""Authenticated API used by the Next.js frontend."""
import json
import time

import jwt
import pymupdf
import pytest
from fastapi.testclient import TestClient

import auth
import main
from workspace import STATUS_READY, write_status

DEV = {"X-Dev-User": "alice"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_MODE", "dev")
    monkeypatch.setattr(auth, "SUPABASE_URL", "")
    return TestClient(main.app)


def pdf_bytes(*pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


# ------------------------------------------------------------------- auth ---

def test_requests_without_identity_are_rejected(client):
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/chat", json={"question": "q"}).status_code == 401


def test_supabase_hs256_token_maps_to_user_workspace(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_MODE", "supabase")
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "s3cret-for-tests-only-0123456789abcdef")
    uid = "8f14e45f-ceea-467a-9575-0d1f1c2b3a4b"
    token = jwt.encode({"sub": uid, "aud": "authenticated", "exp": time.time() + 60},
                       "s3cret-for-tests-only-0123456789abcdef", algorithm="HS256")
    r = TestClient(main.app).get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["workspace"] == f"u-{uid}"


def test_supabase_rejects_bad_expired_and_dev_headers(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_MODE", "supabase")
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "s3cret-for-tests-only-0123456789abcdef")
    c = TestClient(main.app)
    forged = jwt.encode({"sub": "x", "aud": "authenticated"}, "wrong-secret-0123456789abcdefghij", algorithm="HS256")
    expired = jwt.encode({"sub": "x", "aud": "authenticated", "exp": time.time() - 10},
                         "s3cret-for-tests-only-0123456789abcdef", algorithm="HS256")
    assert c.get("/api/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    r = c.get("/api/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401 and "expired" in r.json()["detail"]
    assert c.get("/api/me", headers=DEV).status_code == 401          # dev header ignored in production


def test_supabase_asymmetric_token_via_jwks(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ec
    private = ec.generate_private_key(ec.SECP256R1())
    monkeypatch.setattr(auth, "AUTH_MODE", "supabase")

    class FakeJWKS:
        def get_signing_key_from_jwt(self, token):
            return type("K", (), {"key": private.public_key()})()
    monkeypatch.setattr(auth, "_jwks_client", lambda: FakeJWKS())
    token = jwt.encode({"sub": "abc-123", "aud": "authenticated"}, private, algorithm="ES256")
    assert TestClient(main.app).get("/api/me", headers={"Authorization": f"Bearer {token}"}).json()["workspace"] == "u-abc-123"


def test_legacy_username_routes_are_disabled_in_production(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_MODE", "supabase")
    c = TestClient(main.app)
    assert c.get("/api/status/alice").status_code == 404
    assert c.post("/api/query", json={"username": "alice", "question": "q"}).status_code == 404


def test_dev_mode_refuses_to_run_with_supabase_configured(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_MODE", "dev")
    monkeypatch.setattr(auth, "SUPABASE_URL", "https://example.supabase.co")
    assert TestClient(main.app).get("/api/me", headers=DEV).status_code == 500


# ------------------------------------------------------------ workspace ---

def test_me_empty_workspace(client):
    body = client.get("/api/me", headers=DEV).json()
    assert body["status"] == "not_configured" and body["courses"] == [] and body["has_api_key"] is False


def test_set_key(client):
    assert client.put("/api/me/key", json={"gemini_api_key": "  "}, headers=DEV).status_code == 400
    assert client.put("/api/me/key", json={"gemini_api_key": "AIza-test"}, headers=DEV).status_code == 200
    assert client.get("/api/me", headers=DEV).json()["has_api_key"] is True


def test_upload_lists_and_deletes_pdfs(client):
    r = client.post("/api/me/upload", headers=DEV, data={"course": "Radar"},
                    files=[("files", ("chap 1.pdf", pdf_bytes("Radar basics"), "application/pdf"))])
    assert r.status_code == 200 and r.json()["saved"] == ["Radar/chap 1.pdf"]
    courses = client.get("/api/me", headers=DEV).json()["courses"]
    assert courses == [{"name": "Radar", "files": [{"source": "Radar/chap 1.pdf", "name": "chap 1.pdf"}]}]
    assert client.delete("/api/me/files", params={"source": "Radar/chap 1.pdf"}, headers=DEV).status_code == 200
    assert client.get("/api/me", headers=DEV).json()["courses"] == [{"name": "Radar", "files": []}]


def test_upload_rejects_non_pdfs(client):
    r = client.post("/api/me/upload", headers=DEV, data={"course": "Radar"},
                    files=[("files", ("notes.pdf", b"not really a pdf", "application/pdf"))])
    assert r.status_code == 400
    r = client.post("/api/me/upload", headers=DEV, data={"course": "Radar"},
                    files=[("files", ("evil.exe", b"MZ...", "application/octet-stream"))])
    assert r.status_code == 400


def test_users_cannot_see_each_others_files(client):
    client.post("/api/me/upload", headers=DEV, data={"course": "Radar"},
                files=[("files", ("c1.pdf", pdf_bytes("secret"), "application/pdf"))])
    bob = {"X-Dev-User": "bob"}
    assert client.get("/api/me", headers=bob).json()["courses"] == []
    assert client.get("/api/me/page", params={"source": "Radar/c1.pdf", "page": 1}, headers=bob).status_code == 404


def test_page_image_with_highlight_and_path_traversal(client):
    client.post("/api/me/upload", headers=DEV, data={"course": "Radar"},
                files=[("files", ("c1.pdf", pdf_bytes("Intro", "The radar equation relates received power to range."),
                                  "application/pdf"))])
    plain = client.get("/api/me/page", params={"source": "Radar/c1.pdf", "page": 2}, headers=DEV)
    marked = client.get("/api/me/page", params={"source": "Radar/c1.pdf", "page": 2,
                                                  "highlight": "The radar equation relates received power"}, headers=DEV)
    assert plain.status_code == 200 and plain.headers["content-type"] == "image/png"
    assert plain.headers["x-page-count"] == "2"
    assert marked.content != plain.content                     # highlight drawn
    assert client.get("/api/me/page", params={"source": "Radar/c1.pdf", "page": 9}, headers=DEV).status_code == 404
    for evil in ["../bob/data/x.pdf", "/etc/passwd", "../../workspaces/bob/.env"]:
        assert client.get("/api/me/page", params={"source": evil, "page": 1}, headers=DEV).status_code == 404


def test_index_requires_pdfs(client):
    assert client.post("/api/me/index", headers=DEV).status_code == 400


# ------------------------------------------------------------------ chat ---

def test_chat_end_to_end_with_fake_llm_and_keyword_search(client, monkeypatch, isolated_workspaces):
    from langchain_core.documents import Document
    from search import KeywordIndex
    docs = [Document(page_content="The radar equation relates received power to range.",
                     metadata={"source": "Radar/c1.pdf", "page": 2, "course": "Radar", "chunk_index": 0})]
    (isolated_workspaces / "workspaces" / "alice" / "chroma_db_x").mkdir(parents=True)
    write_status("alice", STATUS_READY, db_dir="chroma_db_x")

    class Store:
        def similarity_search(self, query, k, filter=None):
            return docs

    monkeypatch.setattr(main, "FAKE_LLM", True)
    monkeypatch.setattr(main, "open_vector_store", lambda *a: Store())
    monkeypatch.setattr(main, "safe_keyword_index", lambda path: KeywordIndex(docs))
    import devllm
    monkeypatch.setattr(devllm.time, "sleep", lambda s: None)

    r = client.post("/api/chat", json={"question": "What does the radar equation relate?", "course": "Radar"},
                    headers=DEV)
    events = [json.loads(line) for line in r.text.splitlines() if line]
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert events[0]["type"] == "sources" and events[0]["sources"][0]["page"] == 2
    assert "[1]" in answer and events[-1]["type"] == "done"


def test_chat_without_key_explains_what_to_do(client, isolated_workspaces):
    (isolated_workspaces / "workspaces" / "alice" / "chroma_db_x").mkdir(parents=True)
    write_status("alice", STATUS_READY, db_dir="chroma_db_x")
    r = client.post("/api/chat", json={"question": "q"}, headers=DEV)
    assert r.status_code == 400 and "Settings" in r.json()["detail"]
