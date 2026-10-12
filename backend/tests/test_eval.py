"""The evaluation script: CSV loading, scoring and an end-to-end run with fakes."""
import csv
import json
import os
import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
import run_eval  # noqa: E402


def write_csv(path, rows):
    cols = ["id", "course", "type", "language", "question", "reference_answer",
            "source", "page", "also_ok", "verified", "notes"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


ROWS = [
    {"id": "q01", "course": "Radar", "type": "fact", "language": "en", "question": "What is radar?",
     "reference_answer": "Radio detection and ranging.", "source": "Radar/c1.pdf", "page": "13",
     "also_ok": "Radar/dup.pdf#4; Radar/dup2.pdf#9", "verified": "yes"},
    {"id": "q02", "course": "Radar", "type": "negative", "language": "en", "question": "Kalman?",
     "reference_answer": "NOT IN MATERIALS", "verified": "yes"},
    {"id": "q03", "course": "Radar", "type": "fact", "language": "en", "question": "Unchecked",
     "reference_answer": "x", "source": "Radar/c1.pdf", "page": "1", "verified": ""},
]


def test_load_questions_parses_gold_pages_and_skips_unverified(tmp_path):
    write_csv(tmp_path / "q.csv", ROWS)
    qs = run_eval.load_questions(str(tmp_path / "q.csv"))
    assert [q["id"] for q in qs] == ["q01", "q02"]
    assert qs[0]["gold"] == {("Radar/c1.pdf", 13), ("Radar/dup.pdf", 4), ("Radar/dup2.pdf", 9)}
    assert qs[1]["is_negative"] and qs[1]["gold"] == set()


def doc(source, page):
    return Document(page_content=f"text of {source} p{page}", metadata={"source": source, "page": page})


def test_retrieval_metrics_rank_and_file_hit():
    gold = {("Radar/c1.pdf", 13)}
    m = run_eval.retrieval_metrics([doc("Radar/c1.pdf", 2), doc("Radar/c1.pdf", 13)], gold)
    assert m["hit"] and m["rank"] == 2 and m["file_hit"]
    miss = run_eval.retrieval_metrics([doc("Radar/c1.pdf", 2)], gold)
    assert not miss["hit"] and miss["rank"] is None and miss["file_hit"]


def test_citation_metrics():
    sources = [{"id": 1, "source": "Radar/c1.pdf", "page": 2}, {"id": 2, "source": "Radar/c1.pdf", "page": 13}]
    m = run_eval.citation_metrics("Radar means X [2].", sources, {("Radar/c1.pdf", 13)})
    assert m["cited_any"] and m["cited_gold"]
    assert not run_eval.citation_metrics("No cites.", sources, {("Radar/c1.pdf", 13)})["cited_any"]


def test_parse_json_reply_handles_code_fences():
    assert run_eval.parse_json_reply('```json\n{"score": 2, "reason": "ok"}\n```') == {"score": 2, "reason": "ok"}


def test_with_retry_retries_rate_limits_only(monkeypatch):
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 Resource exhausted")
        return "ok"
    assert run_eval.with_retry(flaky) == "ok" and len(calls) == 3

    def broken():
        raise ValueError("bad request")
    try:
        run_eval.with_retry(broken)
        raise AssertionError("should have raised")
    except ValueError:
        pass


class FakeStore:
    DOCS = [doc("Radar/other.pdf", 1), doc("Radar/c1.pdf", 13)]

    def similarity_search(self, query, k, filter=None):
        return self.DOCS[:k]

    def get(self, include):
        return {"documents": [d.page_content for d in self.DOCS], "metadatas": [d.metadata for d in self.DOCS]}


class FakeLLM:
    """Answers with a citation of source [2]; as a judge, returns canned verdicts."""
    def invoke(self, messages):
        prompt = messages[-1].content
        if "Grade the assistant" in prompt:
            return AIMessage(json.dumps({"score": 2, "reason": "matches"}))
        if "Did the assistant decline" in prompt:
            return AIMessage('{"refused": true, "reason": "said not found"}')
        return AIMessage("Radar is radio detection and ranging [2].")


def test_end_to_end_with_fakes(tmp_path):
    write_csv(tmp_path / "q.csv", ROWS)
    qs = run_eval.load_questions(str(tmp_path / "q.csv"))
    results = run_eval.evaluate(qs, FakeStore(), FakeLLM(), FakeLLM(), k=5, log=lambda *_: None, mode="vector")
    summary = run_eval.summarize(results)

    assert summary["hit_at_k"] == 1.0 and summary["mrr"] == 0.5
    assert summary["answer_score"] == 1.0 and summary["cited_gold"] == 1.0
    assert summary["refusal_rate"] == 1.0

    run_eval.write_report(results, summary, {"timestamp": "t", "model": "fake", "embedding_model": "e", "k": 5},
                          str(tmp_path / "out"))
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| Retrieval hit@5 (right page retrieved) | 100% |" in report
    assert "refused ✅" in report
    data = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert data["results"][0]["citation"]["cited"] == ["Radar/c1.pdf p13"]


def test_retrieval_only_mode_makes_no_llm_calls(tmp_path):
    write_csv(tmp_path / "q.csv", ROWS)
    results = run_eval.evaluate(run_eval.load_questions(str(tmp_path / "q.csv")), FakeStore(), None, None,
                                k=1, log=lambda *_: None, mode="vector")
    summary = run_eval.summarize(results)
    assert summary["hit_at_k"] == 0.0 and summary["file_hit_at_k"] == 0.0
    assert "answer_score" not in summary and "answer" not in results[0]


# The exact error Ali got on the free tier (shortened).
DAILY_QUOTA_ERROR = ("Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
                     "{'error': {'code': 429, 'message': 'You exceeded your current quota... Quota exceeded for "
                     "metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
                     "model: gemini-3.8-flash\\nPlease retry in 11h11m16.067487952s.', 'details': [{'quotaId': "
                     "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}, {'retryDelay': '40276s'}]}}")


def test_daily_quota_is_not_retried():
    calls = []

    def exhausted():
        calls.append(1)
        raise RuntimeError(DAILY_QUOTA_ERROR)
    try:
        run_eval.with_retry(exhausted, sleep=lambda s: None)
        raise AssertionError("should have raised")
    except run_eval.QuotaExhausted:
        pass
    assert len(calls) == 1          # no pointless 5s/10s/20s retries


def test_per_minute_limit_is_still_retried():
    calls = []

    def minute_limit():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED ... Please retry in 7.5s. retryDelay': '7s'")
        return "ok"
    assert run_eval.with_retry(minute_limit, sleep=lambda s: None) == "ok"


class QuotaAfter(FakeLLM):
    """Works for the first n calls, then hits the daily quota."""
    def __init__(self, n):
        self.left = n

    def invoke(self, messages):
        if self.left == 0:
            raise RuntimeError(DAILY_QUOTA_ERROR)
        self.left -= 1
        return super().invoke(messages)


def test_quota_stops_run_cleanly_and_keeps_finished_results(tmp_path):
    rows = [dict(ROWS[0], id=f"q{i:02d}") for i in range(1, 5)]
    write_csv(tmp_path / "q.csv", rows)
    saved = []
    llm = QuotaAfter(3)  # q01: answer+judge, q02: answer, then quota on q02's judge
    results = run_eval.evaluate(run_eval.load_questions(str(tmp_path / "q.csv")), FakeStore(), llm, llm,
                                k=5, log=lambda *_: None, on_result=saved.append, mode="vector")
    assert [r["id"] for r in results] == ["q01"] and saved == results

    config = {"timestamp": "t", "model": "m", "embedding_model": "e", "k": 5, "planned": 4, "user": "ali"}
    run_eval.write_report(results, run_eval.summarize(results), config, str(tmp_path / "out"))
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Partial run:** 1 of 4 questions finished" in report and "--resume" in report


def test_message_text_does_not_call_text_property_strings():
    from rag import message_text

    class StrText(str):
        def __call__(self):
            raise AssertionError("deprecated .text() should not be called")

    class Msg:
        text = StrText("hello")
    assert message_text(Msg()) == "hello"


def test_cli_stops_on_quota_then_resume_finishes_same_run(tmp_path, monkeypatch):
    import main
    from workspace import STATUS_READY, save_user_api_key, write_status

    rows = [dict(ROWS[0], id=f"q{i:02d}") for i in range(1, 4)] + [ROWS[1]]
    rows[-1] = dict(rows[-1], id="q04")
    write_csv(tmp_path / "q.csv", rows)
    save_user_api_key("ali", "KEY")
    (tmp_path / "workspaces" / "ali" / "chroma_db_x").mkdir(parents=True)
    write_status("ali", STATUS_READY, db_dir="chroma_db_x")

    monkeypatch.setattr(run_eval, "BACKEND_DIR", str(tmp_path))
    monkeypatch.setattr(run_eval, "QUESTIONS_CSV", str(tmp_path / "q.csv"))
    monkeypatch.setattr(run_eval, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(main, "open_vector_store", lambda path, *args: FakeStore())
    main.get_keyword_index.cache_clear()
    models = []

    def make_llm(key, model=None):
        models.append(model)
        return llm
    monkeypatch.setattr(main, "make_llm", make_llm)

    llm = QuotaAfter(2)                       # day 1: only q01 fits in the quota
    run_eval.main_cli(["--user", "ali", "--delay", "0"])
    assert models[-1] == "gemini-3.5-flash-lite" and models[-2] == main.LLM_MODEL   # judge on its own model
    run_dir = run_eval.latest_run_dir()
    first = json.loads(open(os.path.join(run_dir, "results.json"), encoding="utf-8").read())
    assert [r["id"] for r in first["results"]] == ["q01"] and first["config"]["planned"] == 4

    llm = QuotaAfter(100)                     # day 2: quota reset
    run_eval.main_cli(["--user", "ali", "--resume", "--delay", "0"])
    assert run_eval.latest_run_dir() == run_dir                     # same run folder, not a new one
    final = json.loads(open(os.path.join(run_dir, "results.json"), encoding="utf-8").read())
    assert [r["id"] for r in final["results"]] == ["q01", "q02", "q03", "q04"]
    assert final["summary"]["refusal_rate"] == 1.0
    assert "Partial run" not in open(os.path.join(run_dir, "report.md"), encoding="utf-8").read()


def test_compare_mode_writes_side_by_side_report(tmp_path, monkeypatch):
    import main
    from workspace import STATUS_READY, write_status

    write_csv(tmp_path / "q.csv", ROWS)
    (tmp_path / "workspaces" / "ali" / "chroma_db_x").mkdir(parents=True)
    write_status("ali", STATUS_READY, db_dir="chroma_db_x")
    monkeypatch.setattr(run_eval, "BACKEND_DIR", str(tmp_path))
    monkeypatch.setattr(run_eval, "QUESTIONS_CSV", str(tmp_path / "q.csv"))
    monkeypatch.setattr(run_eval, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(main, "open_vector_store", lambda path, *args: FakeStore())
    main.get_keyword_index.cache_clear()
    monkeypatch.setattr(main, "make_llm", lambda *a: (_ for _ in ()).throw(AssertionError("no LLM in compare")))

    summaries = run_eval.main_cli(["--user", "ali", "--compare"])
    assert set(summaries) == {"vector", "bm25", "hybrid"}
    assert summaries["vector"]["hit_at_k"] == 1.0
    run_dir = [d for d in os.listdir(tmp_path / "results") if d.endswith("_compare")][0]
    report = open(tmp_path / "results" / run_dir / "report.md", encoding="utf-8").read()
    assert "| Metric | vector | bm25 | hybrid |" in report
    assert run_eval.latest_run_dir() is None        # --resume never picks a comparison folder
