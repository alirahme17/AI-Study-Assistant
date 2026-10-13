"""Hybrid retrieval in Postgres: vector (pgvector) + keyword (full text) + RRF fusion."""
import uuid

import pytest

import db
from providers import FakeEmbedder
from search import keyword_search, rrf_fuse, search, tokenize, vector_search

ALICE, BOB = str(uuid.uuid4()), str(uuid.uuid4())

CORPUS = [
    ("Wireless", "ch3.pdf", 13, "DSDV is a table driven protocol based on Bellman-Ford using hop count"),
    ("Wireless", "ch3.pdf", 25, "AODV discovers routes on demand with RREQ and RREP packets"),
    ("Antenna", "p2.pdf", 10, "La bande passante d'une antenne: transfert d'énergie maximal"),
    ("Radar", "c2.pdf", 22, "Radar range equation with system losses and propagation losses"),
    ("Radar", "c1.pdf", 16, "The radar transmits a pulse and measures the round-trip time"),
]


def add_chunks(user_id, rows, embedder=None):
    embedder = embedder or FakeEmbedder()
    with db.connect() as conn:
        for course, name, page, text in rows:
            doc_id = conn.execute(
                "insert into tt_documents (user_id, course, name, storage_path, status) values (%s, %s, %s, '', 'ready') "
                "on conflict (user_id, course, name) do update set status = 'ready' returning id",
                [user_id, course, name]).fetchone()[0]
            conn.execute(
                "insert into tt_chunks (user_id, document_id, course, page, chunk_index, content, embedding) "
                "values (%s, %s, %s, %s, 0, %s, %s::extensions.vector)",
                [user_id, doc_id, course, page, text, db.vector_literal(embedder.embed_documents([text])[0])])


@pytest.fixture
def corpus(database):
    add_chunks(ALICE, CORPUS)
    add_chunks(BOB, [("Radar", "secret.pdf", 1, "Bob's private radar notes about DSDV")])


def pages(docs):
    return [(d.metadata["source"], d.metadata["page"]) for d in docs]


def test_tokenize_strips_accents_case_and_stopwords():
    assert tokenize("Qu'est-ce que la Bande Passante d'une ÉNERGIE ?") == ["bande", "passante", "energie"]
    assert tokenize("802.11 uses 14 channels") == ["802", "11", "uses", "14", "channels"]


def test_keyword_search_finds_exact_technical_terms(corpus):
    with db.connect() as conn:
        hits = keyword_search(conn, ALICE, "What metric does DSDV use?", 3)
    assert pages(hits)[0] == ("Wireless/ch3.pdf", 13)


def test_keyword_search_ignores_accents_and_ranks_rare_terms_first(corpus):
    with db.connect() as conn:
        hits = keyword_search(conn, ALICE, "Quelle est la bande passante en ENERGIE ?", 5)
        # "radar" appears in two chunks, "pulse" in one: the rarer word decides.
        pulse = keyword_search(conn, ALICE, "radar pulse", 5)
    assert pages(hits)[0] == ("Antenna/p2.pdf", 10)
    assert pages(pulse)[0] == ("Radar/c1.pdf", 16)


def test_search_is_scoped_to_the_user_and_course(corpus):
    with db.connect() as conn:
        alice = keyword_search(conn, ALICE, "DSDV radar", 10)
        bob = keyword_search(conn, BOB, "DSDV", 10)
        radar_only = keyword_search(conn, ALICE, "DSDV radar", 10, course="Radar")
        vector = vector_search(conn, ALICE, FakeEmbedder().embed_query("DSDV"), 10)
    assert "Radar/secret.pdf" not in {s for s, _ in pages(alice) + pages(vector)}
    assert pages(bob) == [("Radar/secret.pdf", 1)]
    assert {d.metadata["course"] for d in radar_only} == {"Radar"}


def test_queries_with_only_stopwords_or_unknown_words_return_nothing(corpus):
    with db.connect() as conn:
        assert keyword_search(conn, ALICE, "what is the", 5) == []
        assert keyword_search(conn, ALICE, "zzzunknownword", 5) == []
        assert keyword_search(conn, ALICE, "it's a 'quoted' & (weird) | query!", 5) == []


def test_vector_search_orders_by_similarity(corpus):
    with db.connect() as conn:
        hits = vector_search(conn, ALICE, FakeEmbedder().embed_query("AODV RREQ RREP routes"), 2)
    assert pages(hits)[0] == ("Wireless/ch3.pdf", 25)


def test_hybrid_combines_both_lists(corpus):
    with db.connect() as conn:
        hits = search(conn, ALICE, FakeEmbedder(), "radar range equation losses", 3, mode="hybrid")
        keyword = search(conn, ALICE, FakeEmbedder(), "radar range equation losses", 3, mode="bm25")
    assert pages(hits)[0] == ("Radar/c2.pdf", 22)
    assert pages(keyword)[0] == ("Radar/c2.pdf", 22)
    with pytest.raises(ValueError):
        with db.connect() as conn:
            search(conn, ALICE, FakeEmbedder(), "x", 3, mode="nope")


def test_rrf_rewards_agreement():
    from langchain_core.documents import Document

    def d(i):
        return Document(page_content=str(i), metadata={"chunk_id": i})
    fused = rrf_fuse([[d(1), d(2), d(3)], [d(3), d(4), d(1)]], k=2)
    assert [x.metadata["chunk_id"] for x in fused] == [1, 3]


def test_hybrid_falls_back_to_keywords_when_embedding_quota_is_used_up(corpus):
    from quota import QuotaExhausted

    class OutOfQuota(FakeEmbedder):
        def embed_query(self, text):
            raise QuotaExhausted("PerDay")
    with db.connect() as conn:
        hits = search(conn, ALICE, OutOfQuota(), "What metric does DSDV use?", 3)
        assert pages(hits)[0] == ("Wireless/ch3.pdf", 13)
        with pytest.raises(QuotaExhausted):
            search(conn, ALICE, OutOfQuota(), "DSDV", 3, fallback=False)
