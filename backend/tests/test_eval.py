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
    def similarity_search(self, query, k, filter=None):
        return [doc("Radar/other.pdf", 1), doc("Radar/c1.pdf", 13)][:k]


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
    results = run_eval.evaluate(qs, FakeStore(), FakeLLM(), FakeLLM(), k=5, log=lambda *_: None)
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
                                k=1, log=lambda *_: None)
    summary = run_eval.summarize(results)
    assert summary["hit_at_k"] == 0.0 and summary["file_hit_at_k"] == 0.0
    assert "answer_score" not in summary and "answer" not in results[0]
