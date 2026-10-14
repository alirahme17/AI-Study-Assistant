"""Turn one PDF into page-aware chunks of text.

Every chunk remembers its page, so answers can cite "chap1.pdf, page 12" and the
page viewer can open it. Pages with no text layer (scans, photographed notes) are
read with OCR when an OCR reader is given.
"""
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import needs_ocr
from quota import QuotaExhausted

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Some PDFs map math/symbol fonts to private-use code points, or to unrelated scripts
# (e.g. "2^16" extracted as Oriya letters). This junk pollutes search and confuses the
# model; the course material is English/French, so these ranges are dropped.
_JUNK = re.compile("[-\U000f0000-\U0010ffffऀ-෿�\x00]")
_SPACES = re.compile(r"[ \t ]{2,}")


def clean_text(text: str) -> str:
    """Remove garbage glyphs and the runs of spaces they leave behind."""
    return _SPACES.sub(" ", _JUNK.sub(" ", text))


@dataclass
class Chunk:
    page: int               # 1-based page number in the PDF
    chunk_index: int        # position of the chunk within its page
    content: str
    ocr: bool = False       # text came from OCR (may contain recognition errors)


@dataclass
class ChunkedPdf:
    pages: int
    chunks: List[Chunk] = field(default_factory=list)
    ocr_pages: int = 0          # scanned pages read with OCR
    ocr_skipped: int = 0        # scanned pages NOT read (no OCR, daily OCR quota, or Gemini errors)
    ocr_errors: int = 0         # ...of which failed because Gemini was unavailable
    ocr_quota_hit: bool = False


def chunk_pdf(data: bytes, ocr=None, on_page: Optional[Callable[[int, int], None]] = None) -> ChunkedPdf:
    """Split a PDF (bytes) into chunks, page by page.

    ocr: an object with read_page(page) -> str (see providers.GeminiOCR), or None.
    on_page(page_number, total) is called before each page (progress reporting).
    If the OCR quota runs out, the remaining scanned pages are skipped and counted.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        result = ChunkedPdf(pages=len(doc))
        for page_number, page in enumerate(doc, start=1):
            if on_page:
                on_page(page_number, len(doc))
            text, from_ocr = page.get_text(), False
            if needs_ocr(page):
                if ocr is not None:
                    try:
                        text, from_ocr = ocr.read_page(page), True
                        result.ocr_pages += 1
                    except QuotaExhausted as e:
                        print(f"OCR daily quota reached, scanned pages skipped for now ({e})")
                        result.ocr_quota_hit = True
                        ocr = None          # stop calling OCR for the rest of this PDF
                    except Exception as e:   # model overloaded (503) even after retries: skip this page only
                        print(f"OCR failed on page {page_number}, skipped for now: {type(e).__name__}: {str(e)[:200]}")
                        result.ocr_errors += 1
                if not from_ocr:
                    result.ocr_skipped += 1  # keep whatever little text the page has
            text = clean_text(text)
            if not text.strip():
                continue                    # blank page, or a scan with nothing readable
            for i, piece in enumerate(splitter.split_text(text)):
                result.chunks.append(Chunk(page=page_number, chunk_index=i, content=piece, ocr=from_ocr))
    return result
