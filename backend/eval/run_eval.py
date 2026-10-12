"""Evaluate the RAG pipeline on eval/questions.csv.

Usage (from the backend folder, with the venv active):

    python eval/run_eval.py --user mhmd123                  # full run: retrieval + answers + judge
    python eval/run_eval.py --user mhmd123 --retrieval-only # no Gemini calls, no API key needed
    python eval/run_eval.py --user mhmd123 --ids q05,q06    # a subset

What is measured for each question (negatives excluded where it makes no sense):

  retrieval   hit@k  : was the gold page (or a duplicate listed in also_ok) among the k retrieved chunks?
              file@k : was at least the right PDF retrieved (any page)?
              rank   : position of the first gold chunk (for MRR)
  citation    cited_gold : did the answer cite a source that is the gold page?
              cited_any  : did the answer contain any [n] citation at all?
  answer      score  : judged by the LLM against the reference answer
                       2 = correct and complete, 1 = partially correct, 0 = wrong / not answered
  negatives   refused : for questions whose topic is NOT in the materials, did the assistant
                        decline instead of answering from general knowledge?

Results are written to eval/results/<timestamp>_<model>/ as results.json and report.md.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

QUESTIONS_CSV = os.path.join(BACKEND_DIR, "eval", "questions.csv")
RESULTS_DIR = os.path.join(BACKEND_DIR, "eval", "results")

JUDGE_PROMPT = """You are grading a study assistant's answer against a reference answer written by the teacher.

Question: {question}

Reference answer: {reference}

Assistant's answer: {answer}

Grade the assistant's answer:
- 2 = correct and covers the key points of the reference (wording may differ; extra correct detail is fine)
- 1 = partially correct: some key points right, but something important is missing or slightly wrong
- 0 = wrong, contradicts the reference, or does not answer (e.g. says it could not find the answer)

Ignore citation markers like [1]. Reply with ONLY a JSON object: {{"score": 0|1|2, "reason": "<one short sentence>"}}"""

REFUSAL_PROMPT = """A study assistant must answer ONLY from the student's course materials. The question below is about
a topic that is NOT in the materials, so the correct behaviour is to say it could not find the answer.

Question: {question}

Assistant's answer: {answer}

Did the assistant decline / say the information is not in the materials (true), or did it answer the
question anyway, e.g. from general knowledge (false)? Reply with ONLY a JSON object:
{{"refused": true|false, "reason": "<one short sentence>"}}"""


# ------------------------------------------------------------------ loading ---

def load_questions(path=QUESTIONS_CSV, ids=None, only_verified=True):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if ids and r["id"] not in ids:
            continue
        if only_verified and r.get("verified", "").strip().lower() != "yes":
            continue
        gold = set()
        if r["source"]:
            gold.add((r["source"], int(r["page"])))
        for alt in filter(None, (a.strip() for a in r.get("also_ok", "").split(";"))):
            src, _, page = alt.rpartition("#")
            gold.add((src, int(page)))
        r["gold"] = gold
        r["is_negative"] = r["type"] == "negative"
        out.append(r)
    return out


# ------------------------------------------------------------------ scoring ---

def doc_location(doc):
    meta = doc.metadata or {}
    return meta.get("source"), meta.get("page")


def retrieval_metrics(docs, gold):
    locations = [doc_location(d) for d in docs]
    rank = next((i for i, loc in enumerate(locations, 1) if loc in gold), None)
    gold_files = {src for src, _ in gold}
    return {
        "hit": rank is not None,
        "rank": rank,
        "file_hit": any(src in gold_files for src, _ in locations),
        "retrieved": [f"{s} p{p}" for s, p in locations],
    }


def citation_metrics(answer, sources, gold):
    from rag import cited_ids
    by_id = {s["id"]: (s["source"], s["page"]) for s in sources}
    cited = [by_id[n] for n in cited_ids(answer) if n in by_id]
    return {"cited_any": bool(cited), "cited_gold": any(c in gold for c in cited),
            "cited": [f"{s} p{p}" for s, p in cited]}


def parse_json_reply(text):
    """Pull the first {...} object out of an LLM reply (models sometimes add ``` fences)."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError(f"No JSON in judge reply: {text!r}")
    return json.loads(match.group(0))


def with_retry(fn, attempts=4, base_delay=5):
    """Retry on rate limits / transient API errors with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            transient = re.search(r"429|rate|quota|exhausted|503|unavailable|timeout|deadline", str(e), re.I)
            if not transient or i == attempts - 1:
                raise
            wait = base_delay * 2 ** i
            print(f"    transient error ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)


# ------------------------------------------------------------------- running ---

def evaluate(questions, db, llm=None, judge=None, k=None, delay=0.0, log=print):
    """Run every question through the same retrieval + prompt the app uses."""
    from langchain_core.messages import HumanMessage
    import main
    from rag import build_messages, message_text, number_sources

    k = k or main.TOP_K
    results = []
    for q in questions:
        log(f"[{q['id']}] {q['question'][:80]}")
        res = {"id": q["id"], "course": q["course"], "type": q["type"], "language": q["language"],
               "question": q["question"], "reference": q["reference_answer"],
               "gold": sorted(f"{s} p{p}" for s, p in q["gold"]), "negative": q["is_negative"]}
        started = time.time()
        _, docs = main.retrieve(llm, db, q["question"], q["course"] or None, [], k=k)
        res["retrieval"] = retrieval_metrics(docs, q["gold"])

        if llm is not None:
            context, sources = number_sources(docs)
            answer = with_retry(lambda: message_text(llm.invoke(build_messages(q["question"], context, []))))
            res["answer"] = answer
            res["latency_s"] = round(time.time() - started, 2)
            if q["is_negative"]:
                verdict = parse_json_reply(message_text(with_retry(lambda: judge.invoke(
                    [HumanMessage(REFUSAL_PROMPT.format(question=q["question"], answer=answer))]))))
                res["refused"] = bool(verdict.get("refused"))
                res["judge_reason"] = verdict.get("reason", "")
            else:
                res["citation"] = citation_metrics(answer, sources, q["gold"])
                verdict = parse_json_reply(message_text(with_retry(lambda: judge.invoke(
                    [HumanMessage(JUDGE_PROMPT.format(question=q["question"], reference=q["reference_answer"],
                                                      answer=answer))]))))
                res["score"] = int(verdict.get("score", 0))
                res["judge_reason"] = verdict.get("reason", "")
            time.sleep(delay)
        results.append(res)
    return results


def summarize(results):
    answerable = [r for r in results if not r["negative"]]
    negatives = [r for r in results if r["negative"]]
    n = len(answerable) or 1
    summary = {
        "questions": len(results),
        "answerable": len(answerable),
        "negatives": len(negatives),
        "hit_at_k": sum(r["retrieval"]["hit"] for r in answerable) / n,
        "file_hit_at_k": sum(r["retrieval"]["file_hit"] for r in answerable) / n,
        "mrr": sum(1 / r["retrieval"]["rank"] for r in answerable if r["retrieval"]["rank"]) / n,
    }
    if answerable and "score" in answerable[0]:
        summary.update({
            "answer_score": sum(r["score"] for r in answerable) / (2 * n),
            "fully_correct": sum(r["score"] == 2 for r in answerable) / n,
            "cited_gold": sum(r["citation"]["cited_gold"] for r in answerable) / n,
            "cited_any": sum(r["citation"]["cited_any"] for r in answerable) / n,
        })
    if negatives and "refused" in negatives[0]:
        summary["refusal_rate"] = sum(r["refused"] for r in negatives) / len(negatives)
    by_group = {}
    for key in ("course", "type", "language"):
        groups = {}
        for r in answerable:
            groups.setdefault(r[key], []).append(r)
        by_group[key] = {
            g: {"n": len(rs), "hit_at_k": sum(r["retrieval"]["hit"] for r in rs) / len(rs),
                **({"answer_score": sum(r["score"] for r in rs) / (2 * len(rs))} if "score" in rs[0] else {})}
            for g, rs in sorted(groups.items())
        }
    summary["by"] = by_group
    return summary


def pct(x):
    return f"{100 * x:.0f}%"


def write_report(results, summary, config, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": config, "summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    full = "answer_score" in summary
    lines = [f"# Evaluation report — {config['timestamp']}", "",
             f"Model `{config['model']}` · embeddings `{config['embedding_model']}` · top-k {config['k']} · "
             f"{summary['answerable']} answerable + {summary['negatives']} negative questions", "",
             "## Summary", "", "| Metric | Value |", "|---|---|",
             f"| Retrieval hit@{config['k']} (right page retrieved) | {pct(summary['hit_at_k'])} |",
             f"| Right PDF retrieved (any page) | {pct(summary['file_hit_at_k'])} |",
             f"| MRR | {summary['mrr']:.2f} |"]
    if full:
        lines += [f"| Answer score (judge, 0–2 scaled) | {pct(summary['answer_score'])} |",
                  f"| Fully correct answers | {pct(summary['fully_correct'])} |",
                  f"| Cited the right page | {pct(summary['cited_gold'])} |",
                  f"| Answers with any citation | {pct(summary['cited_any'])} |"]
    if "refusal_rate" in summary:
        lines.append(f"| Refused out-of-scope questions | {pct(summary['refusal_rate'])} |")

    for key, title in (("course", "By course"), ("type", "By question type"), ("language", "By language")):
        lines += ["", f"## {title}", "", "| Group | n | hit@k |" + (" answer |" if full else ""),
                  "|---|---|---|" + ("---|" if full else "")]
        for g, m in summary["by"][key].items():
            lines.append(f"| {g} | {m['n']} | {pct(m['hit_at_k'])} |" + (f" {pct(m['answer_score'])} |" if full else ""))

    lines += ["", "## Per question", "",
              "| id | course | type | hit | rank |" + (" cited | score |" if full else ""),
              "|---|---|---|---|---|" + ("---|---|" if full else "")]
    for r in results:
        if r["negative"]:
            verdict = ("refused ✅" if r.get("refused") else "answered ❌") if "refused" in r else "—"
            lines.append(f"| {r['id']} | {r['course']} | negative | — | — |" + (f" — | {verdict} |" if full else ""))
            continue
        ret = r["retrieval"]
        row = f"| {r['id']} | {r['course']} | {r['type']} | {'✅' if ret['hit'] else '❌'} | {ret['rank'] or '—'} |"
        if full:
            row += f" {'✅' if r['citation']['cited_gold'] else '❌'} | {r['score']}/2 |"
        lines.append(row)

    failures = [r for r in results if (not r["negative"] and (not r["retrieval"]["hit"] or r.get("score", 2) < 2))
                or (r["negative"] and r.get("refused") is False)]
    if failures:
        lines += ["", "## Failures in detail", ""]
        for r in failures:
            lines += [f"### {r['id']} — {r['question']}", "",
                      f"- **Expected page:** {', '.join(r['gold']) or 'none (out of scope)'}",
                      f"- **Retrieved:** {'; '.join(r['retrieval']['retrieved']) or 'nothing'}"]
            if "answer" in r:
                lines += [f"- **Judge:** {r.get('judge_reason', '')}", "", "> " + r["answer"].replace("\n", "\n> "), ""]
            else:
                lines.append("")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="workspace username whose database and API key to use")
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation and judging")
    parser.add_argument("--ids", help="comma-separated question ids, e.g. q01,q05")
    parser.add_argument("--k", type=int, help="number of chunks to retrieve (default: the app's TOP_K)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between questions (rate limits)")
    parser.add_argument("--include-unverified", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(BACKEND_DIR)  # workspaces/ is relative to the backend folder
    import main
    from workspace import db_path, get_user_api_key

    questions = load_questions(ids=set(args.ids.split(",")) if args.ids else None,
                               only_verified=not args.include_unverified)
    if not questions:
        sys.exit("No questions to run (are they marked verified=yes?).")
    persist_directory = db_path(args.user)
    if persist_directory is None:
        sys.exit(f"No database for user '{args.user}'. Build or re-index it first.")

    llm = judge = None
    if not args.retrieval_only:
        key = get_user_api_key(args.user)
        if not key:
            sys.exit(f"No GEMINI_API_KEY in workspaces/{args.user}/.env (or use --retrieval-only).")
        llm = main.make_llm(key)
        judge = main.make_llm(key)

    print(f"Running {len(questions)} questions against {persist_directory} "
          f"({'retrieval only' if llm is None else 'model ' + main.LLM_MODEL})\n")
    results = evaluate(questions, main.open_vector_store(persist_directory), llm, judge,
                       k=args.k, delay=args.delay if llm else 0)
    summary = summarize(results)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config = {"timestamp": stamp, "user": args.user, "db": os.path.basename(persist_directory),
              "model": main.LLM_MODEL if llm else "none (retrieval only)",
              "embedding_model": main.EMBEDDING_MODEL, "k": args.k or main.TOP_K}
    out_dir = os.path.join(RESULTS_DIR, f"{stamp}_{'retrieval' if llm is None else main.LLM_MODEL}")
    write_report(results, summary, config, out_dir)

    print("\n=== Summary ===")
    print(f"Retrieval hit@{config['k']}: {pct(summary['hit_at_k'])}   right PDF: {pct(summary['file_hit_at_k'])}   "
          f"MRR: {summary['mrr']:.2f}")
    if "answer_score" in summary:
        print(f"Answer score: {pct(summary['answer_score'])}   fully correct: {pct(summary['fully_correct'])}   "
              f"cited right page: {pct(summary['cited_gold'])}")
    if "refusal_rate" in summary:
        print(f"Refused out-of-scope questions: {pct(summary['refusal_rate'])}")
    print(f"\nReport: {os.path.relpath(os.path.join(out_dir, 'report.md'), BACKEND_DIR)}")


if __name__ == "__main__":
    main_cli()
