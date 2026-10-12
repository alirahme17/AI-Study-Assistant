"""Sentence-transformer embeddings, with the query/passage prefixes some models require.

Default for NEW indexes: intfloat/multilingual-e5-small (~118M params, 100 languages).
It replaces all-MiniLM-L6-v2, which is English-only: the evaluation showed English
questions failing to find the French Antenna slides.

Every database records the model that built it (status.json -> "embedding_model"),
and queries always use that same model; databases built before this change used
all-MiniLM-L6-v2. Changing EMBEDDING_MODEL only affects the next (re-)index.
"""
import os
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

LEGACY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# E5 models were trained with these prefixes; without them retrieval quality drops.
_E5_PREFIXES = ("query: ", "passage: ")


def prefixes_for(model_name: str):
    """(query_prefix, document_prefix) for a model."""
    if "e5" in model_name.lower().split("/")[-1]:
        return _E5_PREFIXES
    return "", ""


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings backed by sentence-transformers.

    The model loads lazily on first use, so opening a database just to read its
    chunks (e.g. to build the keyword index) does not load the model.
    """

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.document_prefix = prefixes_for(model_name)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=len(texts) > 200)
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._encode([self.query_prefix + text])[0]


@lru_cache(maxsize=4)
def get_embeddings(model_name: str = DEFAULT_MODEL) -> SentenceTransformerEmbeddings:
    """One instance (and one loaded model) per model name per process."""
    return SentenceTransformerEmbeddings(model_name)
