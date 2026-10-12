"""Citations, numbered sources and conversation memory."""
import json

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import main
import rag
from workspace import STATUS_READY, save_user_api_key, write_status


def doc(text, source="Radar/Course/chap1.pdf", page=1, course="Radar"):
    return Document(page_content=text, metadata={
        "source": source, "file_name": source.split("/")[-1], "course": course, "page": page})


# ------------------------------------------------------------ pure helpers ---

def test_sources_are_numbered_and_same_page_shares_a_number():
    context, sources = rag.number_sources([
        doc("Range equation", page=3),
        doc("More on page 3", page=3),
        doc("Doppler", source="Radar/Course/chap2.pdf", page=7),
    ])
    assert [(s["id"], s["source"], s["page"]) for s in sources] == [
        (1, "Radar/Course/chap1.pdf", 3), (2, "Radar/Course/chap2.pdf", 7)]
    assert "[1] (Radar/Course/chap1.pdf, page 3)" in context
    assert context.count("[1] (") == 2
    assert "[2] (Radar/Course/chap2.pdf, page 7)" in context


def test_old_databases_without_pages_still_produce_sources():
    old = Document(page_content="x", metadata={"source_file": "workspaces/a/data/Radar/old.pdf", "course": "Radar"})
    context, sources = rag.number_sources([old])
    assert sources[0]["source"] == "old.pdf" and sources[0]["page"] is None
    assert "[1] (old.pdf)" in context


def test_cited_ids():
    assert rag.cited_ids("Power falls with R^4 [2]. See also [1, 3] and [2].") == [2, 1, 3]
    assert rag.cited_ids("No citations here.") == []


def test_trim_history_keeps_recent_valid_turns():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    history += [{"role": "system", "content": "ignore me"}, {"role": "assistant", "content": "   "}]
    trimmed = rag.trim_history(history)
    assert [m["content"] for m in trimmed] == ["q4", "q5", "q6", "q7", "q8", "q9"]
    long = rag.trim_history([{"role": "assistant", "content": "a" * 10_000}])
    assert len(long[0]["content"]) == rag.MAX_HISTORY_CHARS


def test_prompt_contains_rules_history_and_sources():
    history = [{"role": "user", "content": "What is a radar?"}, {"role": "assistant", "content": "A sensor [1]."}]
    messages = rag.build_messages("And its range?", "[1] (chap1.pdf, page 3)\nRange...", history)
    assert isinstance(messages[0], SystemMessage) and "cite" in messages[0].content
    assert isinstance(messages[1], HumanMessage) and messages[1].content == "What is a radar?"
    assert isinstance(messages[2], AIMessage)
    assert "[1] (chap1.pdf, page 3)" in messages[-1].content and "And its range?" in messages[-1].content


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


def test_no_history_means_no_extra_llm_call():
    llm = RecordingLLM()
    assert rag.condense_question(llm, "What is a radar?", []) == "What is a radar?"
    assert llm.invoked == []


def test_follow_up_is_rewritten_using_history():
    llm = RecordingLLM()
    history = [{"role": "user", "content": "What is a radar?"}, {"role": "assistant", "content": "A sensor."}]
    assert rag.condense_question(llm, "And its range?", history) == "What is the maximum range of a radar?"
    assert "What is a radar?" in llm.invoked[0][-1].content


def test_rewrite_failure_falls_back_to_original_question():
    class Broken:
        def invoke(self, messages):
            raise RuntimeError("quota")
    assert rag.condense_question(Broken(), "And its range?", [{"role": "user", "content": "x"}]) == "And its range?"


def test_message_text_handles_list_content():
    assert rag.message_text(AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}])) == "Hello world"


# --------------------------------------------------------------- endpoint ---

class FakeStore:
    def __init__(self, docs):
        self.docs, self.queries = docs, []

    def similarity_search(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return self.docs


def _setup(tmp_path, monkeypatch, docs, llm):
    save_user_api_key("alice", "KEY")
    (tmp_path / "workspaces" / "alice" / "chroma_db_x").mkdir(parents=True)
    write_status("alice", STATUS_READY, db_dir="chroma_db_x")
    store = FakeStore(docs)
    monkeypatch.setattr(main, "open_vector_store", lambda path, *args: store)
    monkeypatch.setattr(main, "make_llm", lambda key: llm)
    return store


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_query_streams_sources_then_tokens(isolated_workspaces, monkeypatch):
    llm = RecordingLLM()
    _setup(isolated_workspaces, monkeypatch, [doc("R_max formula", page=12)], llm)

    response = TestClient(main.app).post("/api/query", json={
        "username": "alice", "question": "What is the max range?", "course": "Radar"})
    events = _events(response)

    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert events[0]["sources"][0] == {
        "id": 1, "source": "Radar/Course/chap1.pdf", "file_name": "chap1.pdf",
        "course": "Radar", "page": 12, "ocr": False, "snippet": "R_max formula"}
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Range is R_max [1]."


def test_follow_up_uses_history_for_search_and_answer(isolated_workspaces, monkeypatch):
    llm = RecordingLLM()
    store = _setup(isolated_workspaces, monkeypatch, [doc("R_max formula")], llm)

    TestClient(main.app).post("/api/query", json={
        "username": "alice", "question": "And its range?", "course": "Radar",
        "history": [{"role": "user", "content": "What is a radar?"},
                    {"role": "assistant", "content": "A sensor [1]."}]})

    query, kwargs = store.queries[0]
    assert query == "What is the maximum range of a radar?"        # rewritten for search
    assert kwargs["filter"] == {"course": {"$eq": "Radar"}}
    answer_prompt = llm.streamed[0]
    assert any(isinstance(m, HumanMessage) and m.content == "What is a radar?" for m in answer_prompt)
    assert "And its range?" in answer_prompt[-1].content             # original wording answered


def test_llm_error_is_reported_as_event(isolated_workspaces, monkeypatch):
    class Failing(RecordingLLM):
        def stream(self, messages):
            raise RuntimeError("404 model not found")
            yield
    _setup(isolated_workspaces, monkeypatch, [doc("x")], Failing())
    events = _events(TestClient(main.app).post("/api/query", json={"username": "alice", "question": "q"}))
    assert events[-2]["type"] == "error" and "model not found" in events[-2]["message"]
    assert events[-1]["type"] == "done"


def test_rebuild_reindexes_without_drive(isolated_workspaces, monkeypatch):
    import time
    (isolated_workspaces / "workspaces" / "alice" / "data").mkdir(parents=True)
    monkeypatch.setattr(main, "ingest_drive_data", lambda *a: (_ for _ in ()).throw(AssertionError("Drive used")))

    def fake_build(username, **kwargs):
        (isolated_workspaces / "workspaces" / username / "chroma_db_v2").mkdir()
        return "chroma_db_v2"
    monkeypatch.setattr(main, "build_user_database", fake_build)

    client = TestClient(main.app)
    assert client.post("/api/rebuild/alice").status_code == 200
    for _ in range(100):
        if client.get("/api/status/alice").json()["status"] != "processing":
            break
        time.sleep(0.02)
    assert client.get("/api/status/alice").json()["status"] == "ready"


def test_rebuild_without_downloads_is_404():
    assert TestClient(main.app).post("/api/rebuild/nobody").status_code == 404
