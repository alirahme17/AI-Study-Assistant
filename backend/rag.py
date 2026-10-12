"""RAG helpers: numbered sources, citation prompt, and conversation memory.

Flow for one question:
    1. condense_question(): if there is chat history, rewrite a follow-up like
       "and what about its range?" into a standalone search query.
    2. retrieve chunks with that query.
    3. number_sources(): give each distinct (file, page) a number [1], [2], ...
    4. build_messages(): system rules + recent history + context + question.
       The model is told to cite like [1] after each claim.
    5. cited_ids(): find which [n] the answer actually used.
"""
import os
import re
from typing import Iterable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MAX_HISTORY_MESSAGES = 6        # last 3 question/answer pairs
MAX_HISTORY_CHARS = 1500        # per message, so long answers don't crowd out context
SNIPPET_CHARS = 300

SYSTEM_PROMPT = """You are Teleco-Tutor, a study assistant for a student's own course materials.

Rules:
- Answer ONLY from the numbered sources provided with the question. Do not use outside knowledge.
- After every sentence or claim that uses a source, cite it with its number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, say so plainly (e.g. "I couldn't find this in your course materials.") instead of guessing.
- Be clear and concise. Use formulas and step-by-step explanations when they help a student understand.
- Answer in the same language as the question."""

CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, rewrite the follow-up as a
single standalone question that can be understood without the conversation (resolve words like
"it", "that", "the second one"). Keep the same language. If it is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else."""


def message_text(message) -> str:
    """Text of a LangChain message/chunk across library versions.

    Newer Gemini integrations may return content as a list of parts rather than a
    plain string; `.text` normalises that (a method in langchain-core 0.3, a property in 1.x).
    """
    text = getattr(message, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content or ""


def trim_history(history: Optional[Iterable[dict]]) -> List[dict]:
    """Keep the most recent turns, each truncated, and drop anything malformed."""
    clean = [
        {"role": m["role"], "content": str(m["content"])[:MAX_HISTORY_CHARS]}
        for m in (history or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]
    return clean[-MAX_HISTORY_MESSAGES:]


def _history_messages(history: List[dict]) -> list:
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history]


def condense_question(llm, question: str, history: List[dict]) -> str:
    """Turn a follow-up into a standalone search query. No history -> no LLM call."""
    if not history:
        return question
    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
    try:
        rewritten = message_text(llm.invoke([
            SystemMessage(CONDENSE_PROMPT),
            HumanMessage(f"Conversation:\n{transcript}\n\nFollow-up question: {question}"),
        ])).strip()
    except Exception as e:  # retrieval still works with the raw question
        print(f"Question rewrite failed, using original question: {e}")
        return question
    return rewritten or question


def _source_of(metadata: dict) -> str:
    # Databases built before page-level indexing stored an absolute "source_file".
    return metadata.get("source") or os.path.basename(metadata.get("source_file", "")) or "unknown"


def number_sources(docs) -> tuple:
    """Assign [1], [2], ... to distinct (file, page) pairs.

    Returns (context_text, sources) where sources is a JSON-friendly list:
        {"id", "source", "file_name", "course", "page", "snippet"}
    Two chunks from the same page share one number, so citations stay readable.
    """
    sources, ids, blocks = [], {}, []
    for doc in docs:
        meta = doc.metadata or {}
        source = _source_of(meta)
        page = meta.get("page")
        key = (source, page)
        if key not in ids:
            ids[key] = len(ids) + 1
            sources.append({
                "id": ids[key],
                "source": source,
                "file_name": meta.get("file_name") or os.path.basename(source),
                "course": meta.get("course"),
                "page": page,
                "snippet": doc.page_content[:SNIPPET_CHARS].strip(),
            })
        label = f"{source}, page {page}" if page else source
        blocks.append(f"[{ids[key]}] ({label})\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks), sources


def build_messages(question: str, context: str, history: List[dict]) -> list:
    if context.strip():
        user_turn = f"Sources:\n\n{context}\n\n---\n\nQuestion: {question}"
    else:
        user_turn = f"Sources: (none found)\n\nQuestion: {question}"
    return [SystemMessage(SYSTEM_PROMPT), *_history_messages(history), HumanMessage(user_turn)]


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_ids(answer: str) -> List[int]:
    """Numbers cited in the answer, in order of first use: "[2] ... [1, 3]" -> [2, 1, 3]."""
    seen = []
    for group in _CITATION_RE.findall(answer or ""):
        for n in group.split(","):
            n = int(n)
            if n not in seen:
                seen.append(n)
    return seen
