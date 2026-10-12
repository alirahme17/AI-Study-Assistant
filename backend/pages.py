"""Render a cited PDF page as a PNG, with the cited passage highlighted.

The frontend shows this next to the answer, so a student sees exactly where in
their course material a claim comes from.
"""
import os
import re
from typing import Optional

import pymupdf

from workspace import data_path

RENDER_DPI = 110
HIGHLIGHT_RGB = (1.0, 0.878, 0.4)       # highlighter yellow, matches the UI (#FFE066)
_MAX_PHRASES = 6


def resolve_source(username: str, source: str) -> str:
    """Absolute path of a user's PDF; refuses anything outside their data folder."""
    root = os.path.realpath(data_path(username))
    path = os.path.realpath(os.path.join(root, source))
    if not path.startswith(root + os.sep) or not path.lower().endswith(".pdf"):
        raise ValueError("Invalid source path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(source)
    return path


def _phrases(snippet: str):
    """Short phrases from the cited text to search for on the page.

    Whole-snippet search rarely matches (line breaks, hyphenation, ligatures), so we
    look for several runs of ~6 words and highlight every hit.
    """
    words = re.findall(r"\S+", snippet or "")
    for start in range(0, max(len(words) - 5, 1), 6):
        phrase = " ".join(words[start:start + 6])
        if len(phrase) >= 12:
            yield phrase


def render_page(pdf_path: str, page_number: int, highlight: Optional[str] = None,
                dpi: int = RENDER_DPI) -> bytes:
    """PNG bytes of one page (1-based). Highlights phrases from `highlight` when found."""
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_number <= len(doc):
            raise IndexError(f"Page {page_number} does not exist (the PDF has {len(doc)} pages).")
        page = doc[page_number - 1]
        if highlight:
            for phrase in list(_phrases(highlight))[:_MAX_PHRASES]:
                for quad in page.search_for(phrase, quads=True):
                    annot = page.add_highlight_annot(quad)
                    annot.set_colors(stroke=HIGHLIGHT_RGB)
                    annot.update()
        return page.get_pixmap(dpi=dpi, annots=True).tobytes("png")


def page_count(pdf_path: str) -> int:
    with pymupdf.open(pdf_path) as doc:
        return len(doc)
