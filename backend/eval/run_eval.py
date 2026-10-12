"""Evaluate the RAG pipeline on eval/questions.csv.

Usage (from the backend folder, with the venv active):

    python eval/run_eval.py --user mhmd123                  # full run: retrieval + answers + judge
    python eval/run_eval.py --user mhmd123 --retrieval-only # no Gemini calls, no API key needed
    python eval/run_eval.py --user mhmd123 --ids q05,q06    # a subset
    python eval/run_eval.py --user mhmd123 --resume         # continue the latest unfinished run
    python eval/run_eval.py --user mhmd123 --compare        # retrieval only: vector vs bm25 vs hybrid

Free-tier Gemini keys have a small DAILY quota per model. The judge therefore uses a different
model (--judge-model) so its calls come out of a separate quota, and when a daily quota runs out
the run stops cleanly, keeps everything finished so far, and can be continued later with --resume.

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

def load_questions(path=None, ids=None, only_verified=True):
    with open(path or QUESTIONS_CSV, encoding="utf-8-sig", newline="") as f:
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


class QuotaExhausted(Exception):
    """A daily (or otherwise long) quota ran out: retrying now is pointless."""


def _is_long_quota(message: str) -> bool:
    if re.search(r"PerDay|per day|daily", message, re.I):
        return True
    # "Please retry in 11h11m16s" / "retryDelay': '40276s'" -> anything over 2 minutes
    hours = re.search(r"retry in (\d+)h", message)
    seconds = re.search(r"retryDelay'?:\s*'(\d+)s", message)
    return bool(hours) or bool(seconds and int(seconds.group(1)) > 120)


def with_retry(fn, attempts=4, base_delay=5, sleep=None):
    """Retry on per-minute rate limits / transient errors with exponential backoff.

    Daily quota errors raise QuotaExhausted immediately instead of retrying.
    """
    sleep = sleep or time.sleep
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            message = str(e)
            if re.search(r"429|RESOURCE_EXHAUSTED|quota", message, re.I) and _is_long_quota(message):
                raise QuotaExhausted(message.split("\n")[0][:300]) from e
            transient = re.search(r"429|rate|quota|exhausted|503|unavailable|timeout|deadline", message, re.I)
            if not transient or i == attempts - 1:
                raise
            wait = base_delay * 2 ** i
            print(f"    transient error ({type(e).__name__}), retrying in {wait}s...")
            sleep(wait)


# ------------------------------------------------------------------- running ---

def evaluate(questions, db, llm=None, judge=None, k=None, delay=0.0, log=print, on_result=None,
             keyword_index=None, mode=None):
    """Run every question through the same retrieval + prompt the app uses.

    Calls on_result(res) after each question (used to save progress). Stops early,
    returning what is finished, if a daily API quota runs out.
    """
    from langchain_core.messages import HumanMessage
    import main
    from rag import build_messages, message_text, number_sources

    k = k or main.TOP_K
    results = []
    for q in questions:
        log(f"[{q['id']}] {q['question'][:80]}")
        try:
            res = _evaluate_one(q, db, llm, judge, k, main, build_messages, message_text, number_sources,
                                HumanMessage, keyword_index, mode)
        except QuotaExhausted as e:
            log(f"\n!! Daily API quota exhausted, stopping early: {e}")
            break
        results.append(res)
        if on_result:
            on_result(res)
        if llm is not None:
            time.sleep(delay)
    return results


def _evaluate_one(q, db, llm, judge, k, main, build_messages, message_text, number_sources, HumanMessage,
                  keyword_index=None, mode=None):
    """Retrieve, answer and judge one question; raises QuotaExhausted on a daily quota."""
    res = {"id": q["id"], "course": q["course"], "type": q["type"], "language": q["language"],
           "question": q["question"], "reference": q["reference_answer"],
           "gold": sorted(f"{s} p{p}" for s, p in q["gold"]), "negative": q["is_negative"]}
    started = time.time()
    _, docs = main.retrieve(llm, db, q["question"], q["course"] or None, [], k=k,
                            keyword_index=keyword_index, mode=mode)
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
    return res


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
    lines = [f"# Evaluation report — {config['timestamp']}", ""]
    if config.get("planned") and config["planned"] > len(results):
        lines += [f"> ⚠️ **Partial run:** {len(results)} of {config['planned']} questions finished "
                  f"(API quota ran out). Continue with `python eval/run_eval.py --user {config.get('user')} --resume`.", ""]
    lines += [
             f"Answers `{config['model']}` · judge `{config.get('judge_model') or '—'}` · embeddings `{config['embedding_model']}` · retrieval `{config.get('retrieval', 'vector')}` · top-k {config['k']} · "
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


def compare_modes(questions, db, keyword_index, k, persist_directory, embedding_model):
    """Retrieval-only comparison of the three modes; writes eval/results/<stamp>_compare/report.md."""
    import main
    modes = ["vector", "bm25", "hybrid"]
    runs = {m: evaluate(questions, db, None, None, k=k, log=lambda *_: None, keyword_index=keyword_index, mode=m)
            for m in modes}
    summaries = {m: summarize(r) for m, r in runs.items()}
    answerable = [q for q in questions if not q["is_negative"]]

    lines = [f"# Retrieval comparison — {datetime.now().strftime('%Y%m%d-%H%M%S')}", "",
             f"Embeddings `{embedding_model}` · top-k {k} · {len(answerable)} answerable questions · "
             f"database `{os.path.basename(persist_directory)}`", "",
             "| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes),
             f"| hit@{k} (right page) | " + " | ".join(pct(summaries[m]["hit_at_k"]) for m in modes) + " |",
             "| right PDF | " + " | ".join(pct(summaries[m]["file_hit_at_k"]) for m in modes) + " |",
             "| MRR | " + " | ".join(f"{summaries[m]['mrr']:.2f}" for m in modes) + " |", ""]
    for key, title in (("language", "hit@k by language"), ("course", "hit@k by course")):
        groups = sorted(summaries[modes[0]]["by"][key])
        lines += [f"## {title}", "", "| Group | n | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
        for g in groups:
            lines.append(f"| {g} | {summaries[modes[0]]['by'][key][g]['n']} | "
                         + " | ".join(pct(summaries[m]["by"][key][g]["hit_at_k"]) for m in modes) + " |")
        lines.append("")
    lines += ["## Per question (rank of the right page, — = not in top k)", "",
              "| id | lang | " + " | ".join(modes) + " | question |", "|---|---|" + "---|" * len(modes) + "---|"]
    by_id = {m: {r["id"]: r for r in runs[m]} for m in modes}
    for q in answerable:
        lines.append(f"| {q['id']} | {q['language']} | "
                     + " | ".join(str(by_id[m][q['id']]["retrieval"]["rank"] or "—") for m in modes)
                     + f" | {q['question'][:70]} |")

    out_dir = os.path.join(RESULTS_DIR, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_compare")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"k": k, "summaries": summaries, "results": runs}, f, indent=2, ensure_ascii=False)

    print(f"{'':12}" + "".join(f"{m:>10}" for m in modes))
    print(f"{'hit@' + str(k):12}" + "".join(f"{pct(summaries[m]['hit_at_k']):>10}" for m in modes))
    print(f"{'right PDF':12}" + "".join(f"{pct(summaries[m]['file_hit_at_k']):>10}" for m in modes))
    print(f"{'MRR':12}" + "".join(f"{summaries[m]['mrr']:>10.2f}" for m in modes))
    print(f"\nReport: {os.path.relpath(os.path.join(out_dir, 'report.md'), BACKEND_DIR)}")
    return summaries


def latest_run_dir():
    if not os.path.isdir(RESULTS_DIR):
        return None
    runs = sorted(d for d in os.listdir(RESULTS_DIR)
                  if not d.endswith("_compare") and os.path.isfile(os.path.join(RESULTS_DIR, d, "results.json")))
    return os.path.join(RESULTS_DIR, runs[-1]) if runs else None


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="workspace username whose database and API key to use")
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation and judging")
    parser.add_argument("--ids", help="comma-separated question ids, e.g. q01,q05")
    parser.add_argument("--k", type=int, help="number of chunks to retrieve (default: the app's TOP_K)")
    parser.add_argument("--model", help="model that writes the answers (default: the app's model, GEMINI_MODEL)")
    parser.add_argument("--judge-model", default=os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite"),
                        help="model that grades the answers; a different model uses a separate free-tier quota")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="RUN_DIR",
                        help="continue an unfinished run (default: the latest one), skipping finished questions")
    parser.add_argument("--retrieval", choices=["vector", "bm25", "hybrid"],
                        help="retrieval mode (default: the app's RETRIEVAL_MODE)")
    parser.add_argument("--compare", action="store_true",
                        help="retrieval only: run vector, bm25 and hybrid and compare them (no API calls)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between questions (rate limits)")
    parser.add_argument("--include-unverified", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(BACKEND_DIR)  # workspaces/ is relative to the backend folder
    import main
    from workspace import db_path, embedding_model_for, get_user_api_key

    questions = load_questions(ids=set(args.ids.split(",")) if args.ids else None,
                               only_verified=not args.include_unverified)
    if not questions:
        sys.exit("No questions to run (are they marked verified=yes?).")
    persist_directory = db_path(args.user)
    if persist_directory is None:
        sys.exit(f"No database for user '{args.user}'. Build or re-index it first.")

    embedding_model = embedding_model_for(args.user)
    db = main.open_vector_store(persist_directory, embedding_model)
    mode = args.retrieval or main.RETRIEVAL_MODE
    keyword_index = None if mode == "vector" and not args.compare else main.get_keyword_index(persist_directory)

    if args.compare:
        return compare_modes(questions, db, keyword_index, args.k or main.TOP_K, persist_directory, embedding_model)

    answer_model = args.model or main.LLM_MODEL
    llm = judge = None
    if not args.retrieval_only:
        key = get_user_api_key(args.user)
        if not key:
            sys.exit(f"No GEMINI_API_KEY in workspaces/{args.user}/.env (or use --retrieval-only).")
        llm = main.make_llm(key, answer_model)
        judge = main.make_llm(key, args.judge_model)

    # Resume: reuse the previous run's folder and skip questions it already finished.
    previous = []
    if args.resume:
        out_dir = latest_run_dir() if args.resume == "latest" else os.path.abspath(args.resume)
        if not out_dir:
            sys.exit("Nothing to resume: no previous run in eval/results/.")
        with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
            saved = json.load(f)
        config = saved["config"]
        previous = saved["results"]
        done = {r["id"] for r in previous}
        questions = [q for q in questions if q["id"] not in done]
        print(f"Resuming {os.path.basename(out_dir)}: {len(done)} done, {len(questions)} to go.")
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config = {"timestamp": stamp, "user": args.user, "db": os.path.basename(persist_directory),
                  "model": answer_model if llm else "none (retrieval only)",
                  "judge_model": args.judge_model if llm else None,
                  "embedding_model": embedding_model, "retrieval": mode, "k": args.k or main.TOP_K}
        out_dir = os.path.join(RESULTS_DIR, f"{stamp}_{answer_model if llm else 'retrieval'}")
    config["planned"] = len(previous) + len(questions)

    results = list(previous)

    def save(res=None):
        if res is not None:
            results.append(res)
        write_report(results, summarize(results), config, out_dir)

    print(f"Running {len(questions)} questions against {persist_directory} "
          f"(retrieval: {config.get('retrieval', mode)}; "
          f"{'no answers' if llm is None else f'answers: {answer_model}, judge: {args.judge_model}'})\n")
    evaluate(questions, db, llm, judge, k=config["k"], delay=args.delay if llm else 0, on_result=save,
             keyword_index=keyword_index, mode=config.get("retrieval", mode))
    save()
    summary = summarize(results)

    print("\n=== Summary ===")
    if len(results) < config["planned"]:
        print(f"PARTIAL: {len(results)}/{config['planned']} questions finished. "
              f"Run again later with --resume to finish.")
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
