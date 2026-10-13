"""Hybrid retrieval in Postgres: vector search (pgvector) + keyword search (full text),
fused with Reciprocal Rank Fusion.

Why both: embeddings capture meaning ("how far can a radar see" ~ "maximum range"),
keywords catch exact technical tokens ("DSDV", "RRTS", "PC1", "S11") that embeddings
blur. RRF merges the two ranked lists without calibrating their score scales.

Keyword scoring approximates BM25: each query term's match score (ts_rank, length
normalised) is weighted by the term's IDF in the user's own chunks, so rare,
specific words count more than words found everywhere.
"""
import math
import re
import unicodedata
from typing import List, Optional

from langchain_core.documents import Document

from db import vector_literal

RRF_K = 60                 # standard RRF constant: dampens the weight of top ranks
CANDIDATES_PER_LIST = 4    # each list contributes k * this many candidates to the fusion
MAX_QUERY_TERMS = 24

# Small EN + FR stop-word list: words that match everywhere and only add noise.
STOPWORDS = set("""
a an and are as at be by can do does for from has have how in is it its of on or that the
their them then there these this to was were what when where which who why will with
au aux avec ce ces dans de des du elle en est et il ils la le les leur mais ne ou par pas
pour qu que qui sa se ses son sont sur un une est-ce quel quelle quelles quels
""".split())

_COLUMNS = """c.id, c.content, c.page, c.chunk_index, c.ocr, c.course, d.name, d.id as document_id"""


def tokenize(text: str) -> List[str]:
    """Lowercase, strip accents (so 'énergie' matches 'energie'), split into words."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def _query_terms(query: str) -> List[str]:
    seen = []
    for t in tokenize(query):
        if t not in seen:
            seen.append(t)
    return seen[:MAX_QUERY_TERMS]


def _to_doc(row) -> Document:
    chunk_id, content, page, chunk_index, ocr, course, name, document_id = row[:8]
    source = f"{course}/{name}"
    return Document(page_content=content, metadata={
        "chunk_id": chunk_id, "source": source, "file_name": name.rsplit("/", 1)[-1],
        "course": course, "page": page, "chunk_index": chunk_index, "ocr": ocr,
        "document_id": str(document_id),
    })


def _scope(user_id, course: Optional[str]):
    sql, params = "c.user_id = %s", [user_id]
    if course:
        sql += " and c.course = %s"
        params.append(course)
    return sql, params


def vector_search(conn, user_id, query_vector: List[float], k: int, course: Optional[str] = None) -> List[Document]:
    where, params = _scope(user_id, course)
    vec = vector_literal(query_vector)
    rows = conn.execute(
        f"select {_COLUMNS} from tt_chunks c join tt_documents d on d.id = c.document_id "
        f"where {where} order by c.embedding OPERATOR(extensions.<=>) %s::extensions.vector limit %s",
        [*params, vec, k]).fetchall()
    return [_to_doc(r) for r in rows]


def keyword_search(conn, user_id, query: str, k: int, course: Optional[str] = None) -> List[Document]:
    terms = _query_terms(query)
    if not terms:
        return []
    where, params = _scope(user_id, course)

    # Document frequency of each term among this user's chunks -> IDF weights.
    stats = conn.execute(
        "select count(*), " + ", ".join("count(*) filter (where c.fts @@ to_tsquery('simple', %s))" for _ in terms)
        + f" from tt_chunks c where {where}", [*terms, *params]).fetchone()
    total, dfs = stats[0], stats[1:]
    weighted = [(t, math.log(1 + (total - df + 0.5) / (df + 0.5))) for t, df in zip(terms, dfs) if df > 0]
    if not weighted:
        return []

    score = " + ".join("%s * ts_rank(c.fts, to_tsquery('simple', %s), 1)" for _ in weighted)
    any_term = " | ".join(t for t, _ in weighted)       # tokens are [a-z0-9]+: safe in a tsquery
    score_params = [p for t, idf in weighted for p in (idf, t)]
    rows = conn.execute(
        f"select {_COLUMNS}, ({score}) as score from tt_chunks c join tt_documents d on d.id = c.document_id "
        f"where {where} and c.fts @@ to_tsquery('simple', %s) order by score desc, c.id limit %s",
        [*score_params, *params, any_term, k]).fetchall()
    return [_to_doc(r) for r in rows]


def rrf_fuse(ranked_lists: List[List[Document]], k: int, rrf_k: int = RRF_K) -> List[Document]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank)."""
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata["chunk_id"]
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in best]


def search(conn, user_id, embedder, query: str, k: int, course: Optional[str] = None,
           mode: str = "hybrid", fallback: bool = True) -> List[Document]:
    """Retrieve k chunks of this user's documents with mode "vector", "keyword" or "hybrid".

    fallback: if the embedding API's daily quota is used up, answer with keyword search
    alone instead of failing (the evaluation turns this off to measure each mode honestly).
    """
    from quota import QuotaExhausted
    if mode == "bm25":
        mode = "keyword"          # name used by older evaluation reports
    if mode not in ("vector", "keyword", "hybrid"):
        raise ValueError(f"Unknown retrieval mode: {mode!r}")
    query_vector = None
    if mode != "keyword":
        try:
            query_vector = embedder.embed_query(query)
        except QuotaExhausted:
            if not fallback:
                raise
            print("Embedding quota used up: keyword search only for this question.")
            mode = "keyword"
    n = k * CANDIDATES_PER_LIST if mode == "hybrid" else k
    vector_hits = vector_search(conn, user_id, query_vector, n, course) if mode != "keyword" else []
    keyword_hits = keyword_search(conn, user_id, query, n, course) if mode != "vector" else []
    if mode == "vector":
        return vector_hits
    if mode == "keyword":
        return keyword_hits
    return rrf_fuse([vector_hits, keyword_hits], k)
