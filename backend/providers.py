"""AI providers behind small interfaces, so the hosted app needs no local models.

Embeddings (EMBEDDINGS=gemini | fake)
    GeminiEmbedder  gemini-embedding-2 via the Gemini API, 768 dimensions. Multilingual,
                    so English questions find French slides and vice versa.
    FakeEmbedder    hashed bag of words, offline. Tests and demos only: keyword-ish
                    similarity, no understanding of meaning.

OCR (OCR=gemini | none)
    GeminiOCR       a cheap Gemini vision model reads scanned pages - including
                    handwriting, which classic OCR could not read.

The free host has 512 MB RAM, which can't hold a local embedding model next to the
app, and the Gemini API is free within daily limits. Each user's own API key is used,
so every user has their own quota.
"""
import hashlib
import math
import re
import unicodedata
from typing import List, Optional

import settings
from quota import with_retry

EMBED_BATCH = 50          # texts per embedding request (fewer requests -> less rate limiting)
OCR_DPI = 150             # enough for a vision model to read slides and handwriting
MIN_TEXT_CHARS = 30       # a page with less embedded text than this is treated as scanned

OCR_PROMPT = """Transcribe all the text on this page of a student's course material.
- Keep the original language (English or French). Do not translate or summarise.
- Include handwriting, labels on figures, and table contents (as simple rows).
- Write formulas in LaTeX ($...$).
- If there is no readable text at all, reply with exactly: (no text)
Return only the transcription."""


def _l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


# ----------------------------------------------------------------------------- embeddings

class GeminiEmbedder:
    """gemini-embedding-2, truncated to 768 dimensions (Matryoshka) and re-normalised.

    gemini-embedding-2 has no task_type parameter: the task goes into the text itself,
    with different prefixes for queries and documents.
    """

    def __init__(self, api_key: str, model: Optional[str] = None, dims: int = settings.EMBEDDING_DIMS,
                 client=None):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self.client = client
        self.model = model or settings.EMBEDDING_MODEL
        self.dims = dims

    def _embed(self, texts: List[str]) -> List[List[float]]:
        from google.genai import types
        out = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            # One Content per text: a plain list of strings would be embedded as ONE input.
            contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
            result = with_retry(lambda: self.client.models.embed_content(
                model=self.model, contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=self.dims)))
            if len(result.embeddings) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(result.embeddings)}.")
            out.extend(_l2_normalize(list(e.values)) for e in result.embeddings)
        return out

    def embed_documents(self, texts: List[str], title: str = "none") -> List[List[float]]:
        return self._embed([f"title: {title} | text: {t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._embed([f"task: search result | query: {text}"])[0]


def _words(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]+", text)


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors (offline tests and demos only)."""

    model = "fake-hash-768"

    def __init__(self, dims: int = settings.EMBEDDING_DIMS):
        self.dims = dims

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dims
        for word in _words(text):
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            vector[h % self.dims] += 1.0 if (h >> 32) & 1 else -1.0
        if not any(vector):
            vector[0] = 1.0
        return _l2_normalize(vector)

    def embed_documents(self, texts: List[str], title: str = "none") -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def make_embedder(api_key: Optional[str]):
    if settings.EMBEDDINGS == "fake":
        return FakeEmbedder()
    if not api_key:
        raise RuntimeError("Add your Gemini API key in Settings first.")
    return GeminiEmbedder(api_key)


# ----------------------------------------------------------------------------- OCR

def needs_ocr(page) -> bool:
    """No usable text layer, but there is an image to read (a scan or a photo)."""
    return len(page.get_text().strip()) < MIN_TEXT_CHARS and bool(page.get_images())


class GeminiOCR:
    def __init__(self, api_key: str, model: Optional[str] = None, client=None):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self.client = client
        self.model = model or settings.OCR_MODEL

    def read_page(self, page) -> str:
        """Text of a PyMuPDF page, read from its rendered image."""
        from google.genai import types
        png = page.get_pixmap(dpi=OCR_DPI).tobytes("png")
        result = with_retry(lambda: self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=png, mime_type="image/png"), OCR_PROMPT],
            config=types.GenerateContentConfig(temperature=0)))
        text = (result.text or "").strip()
        return "" if text == "(no text)" else text


def make_ocr(api_key: Optional[str]):
    """An OCR reader, or None when OCR is disabled or there is no key."""
    if settings.OCR != "gemini" or not api_key:
        return None
    return GeminiOCR(api_key)
