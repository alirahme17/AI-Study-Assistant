"""Keyword (BM25) search and hybrid fusion with the vector search.

Why: the embedding model (all-MiniLM-L6-v2) captures meaning but is weak on exact
technical tokens ("DSDV", "RRTS", "PC1", "S11") and on French text. BM25 matches
words exactly. Reciprocal Rank Fusion (RRF) merges both ranked lists without
having to calibrate their very different score scales.

The keyword index is built from the chunks already stored in Chroma, so existing
databases get hybrid search without re-indexing.
"""
import re
import unicodedata
from typing import Iterable, List, Optional

from langchain_core.documents import Document

RRF_K = 60            # standard RRF constant: dampens the weight of top ranks
CANDIDATES_PER_LIST = 4   # each list contributes TOP_K * this many candidates to the fusion

# Small EN + FR stop-word list: words that match everywhere and only add noise to BM25.
STOPWORDS = set("""
a an and are as at be by can do does for from has have how in is it its of on or that the
their them then there these this to was were what when where which who why will with
au aux avec ce ces dans de des du elle en est et il ils la le les leur mais ne ou par pas
pour qu que qui sa se ses son sont sur un une est-ce quel quelle quelles quels
""".split())


def tokenize(text: str) -> List[str]:
    """Lowercase, strip accents (so 'énergie' matches 'energie'), split into words."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def doc_key(doc: Document) -> tuple:
    """Identity of a chunk across the two searches."""
    meta = doc.metadata or {}
    return (meta.get("source") or meta.get("source_file"), meta.get("page"), meta.get("chunk_index"),
            hash(doc.page_content))


class KeywordIndex:
    """BM25 over a fixed set of chunks, with optional course filtering."""

    def __init__(self, docs: Iterable[Document]):
        from rank_bm25 import BM25Okapi
        self.docs = list(docs)
        tokenized = [tokenize(d.page_content) for d in self.docs]
        # BM25Okapi cannot handle an empty corpus or all-empty documents.
        self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    @classmethod
    def from_chroma(cls, db) -> "KeywordIndex":
        data = db.get(include=["documents", "metadatas"])
        return cls(Document(page_content=text or "", metadata=meta or {})
                   for text, meta in zip(data["documents"], data["metadatas"]))

    def search(self, query: str, k: int, course: Optional[str] = None) -> List[Document]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if course and (self.docs[i].metadata or {}).get("course") != course:
                continue
            out.append(self.docs[i])
            if len(out) == k:
                break
        return out


def rrf_fuse(ranked_lists: List[List[Document]], k: int, rrf_k: int = RRF_K) -> List[Document]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank)."""
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc_key(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in best]


def search(db, keyword_index: Optional[KeywordIndex], query: str, k: int,
           course: Optional[str] = None, mode: str = "hybrid") -> List[Document]:
    """Retrieve k chunks with mode "vector", "bm25" or "hybrid"."""
    if mode not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"Unknown retrieval mode: {mode!r}")
    if mode != "vector" and keyword_index is None:
        mode = "vector"  # no keyword index available: degrade gracefully

    n_candidates = k * CANDIDATES_PER_LIST if mode == "hybrid" else k
    vector_hits, keyword_hits = [], []
    if mode in ("vector", "hybrid"):
        kwargs = {"k": n_candidates}
        if course:
            kwargs["filter"] = {"course": {"$eq": course}}
        vector_hits = db.similarity_search(query=query, **kwargs)
    if mode in ("bm25", "hybrid"):
        keyword_hits = keyword_index.search(query, n_candidates, course)

    if mode == "vector":
        return vector_hits
    if mode == "bm25":
        return keyword_hits
    return rrf_fuse([vector_hits, keyword_hits], k)
