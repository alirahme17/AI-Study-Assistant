"""Keyword search (BM25), rank fusion and hybrid retrieval."""
from langchain_core.documents import Document

import main
from search import KeywordIndex, rrf_fuse, search, tokenize


def doc(text, source="Radar/c1.pdf", page=1, course="Radar", chunk=0):
    return Document(page_content=text, metadata={"source": source, "page": page, "course": course, "chunk_index": chunk})


CORPUS = [
    doc("DSDV is a table driven protocol based on Bellman-Ford using hop count", "WN/ch3.pdf", 13, "Wireless"),
    doc("AODV discovers routes on demand with RREQ and RREP packets", "WN/ch3.pdf", 25, "Wireless"),
    doc("La bande passante d'une antenne: transfert d'énergie maximal", "Ant/p2.pdf", 10, "Antenna"),
    doc("Radar range equation with system losses and propagation losses", "Radar/c2.pdf", 22, "Radar"),
    doc("The radar transmits a pulse and measures the round-trip time", "Radar/c1.pdf", 16, "Radar"),
]


def test_tokenize_strips_accents_case_and_stopwords():
    assert tokenize("Qu'est-ce que la Bande Passante d'une ÉNERGIE ?") == ["bande", "passante", "energie"]
    assert tokenize("802.11 uses 14 channels") == ["802", "11", "uses", "14", "channels"]


def test_bm25_finds_exact_technical_terms():
    index = KeywordIndex(CORPUS)
    assert index.search("What metric does DSDV use?", 1)[0].metadata["page"] == 13
    assert index.search("energie bande passante", 1)[0].metadata["source"] == "Ant/p2.pdf"   # accent-insensitive


def test_bm25_respects_course_filter_and_skips_zero_scores():
    index = KeywordIndex(CORPUS)
    hits = index.search("radar losses", 5, course="Radar")
    assert hits and all(h.metadata["course"] == "Radar" for h in hits)
    assert index.search("radar losses", 5, course="Wireless") == []
    assert index.search("quantum chromodynamics", 5) == []


def test_empty_index_and_empty_query():
    assert KeywordIndex([]).search("anything", 3) == []
    assert KeywordIndex(CORPUS).search("the of and", 3) == []    # only stop-words


def test_rrf_rewards_documents_ranked_well_in_both_lists():
    a, b, c, d = CORPUS[:4]
    fused = rrf_fuse([[a, b, c], [c, d, a]], k=2)
    assert fused == [a, c]          # a: ranks 1 and 3, c: ranks 3 and 1 -> beat b and d (one list only)


class FakeVectorStore:
    def __init__(self, docs):
        self.docs, self.calls = docs, []

    def similarity_search(self, query, k, filter=None):
        self.calls.append((query, k, filter))
        return self.docs[:k]

    def get(self, include):
        return {"documents": [d.page_content for d in CORPUS], "metadatas": [d.metadata for d in CORPUS]}


def test_hybrid_combines_vector_and_keyword_results():
    store = FakeVectorStore([CORPUS[4], CORPUS[1]])          # vector search misses the DSDV page
    index = KeywordIndex.from_chroma(store)
    hits = search(store, index, "Which metric does DSDV use?", k=2, course=None, mode="hybrid")
    assert CORPUS[0] in hits                                   # recovered by BM25
    assert store.calls[0][1] == 2 * 4                          # vector search fetched extra candidates


def test_modes_and_fallback_without_keyword_index():
    store = FakeVectorStore([CORPUS[4]])
    assert search(store, None, "DSDV", 1, mode="hybrid") == [CORPUS[4]]      # degrades to vector
    assert search(store, KeywordIndex(CORPUS), "DSDV", 1, mode="bm25") == [CORPUS[0]]
    assert store.calls[-1][2] is None
    search(store, None, "x", 1, course="Radar", mode="vector")
    assert store.calls[-1][2] == {"course": {"$eq": "Radar"}}


def test_safe_keyword_index_falls_back_to_none(monkeypatch):
    def broken(path):
        raise RuntimeError("corrupt db")
    main.get_keyword_index.cache_clear()
    monkeypatch.setattr(main, "open_vector_store", broken)
    assert main.safe_keyword_index("/nowhere") is None
