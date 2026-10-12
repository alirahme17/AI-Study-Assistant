import os
import re
from typing import Callable, Optional

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ocr import OcrCache, needs_ocr, ocr_available, ocr_page
from workspace import data_path, workspace_path

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Some PDFs map math/symbol fonts to private-use code points, or to unrelated scripts
# (e.g. "2^16" extracted as Oriya letters). This junk pollutes search and confuses the
# model; the course material is English/French, so these ranges are dropped.
_JUNK = re.compile("[\ue000-\uf8ff\U000f0000-\U0010ffff\u0900-\u0dff\ufffd]")
_SPACES = re.compile(r"[ \t\u00a0]{2,}")


def clean_text(text: str) -> str:
    """Remove garbage glyphs and the runs of spaces they leave behind."""
    return _SPACES.sub(" ", _JUNK.sub(" ", text))


def _course_of(relative_path: str) -> str:
    parts = relative_path.split(os.sep)
    return parts[0] if len(parts) > 1 else "Unknown"


def process_pdf(file_path: str, relative_path: str, ocr_cache: Optional[OcrCache] = None,
                use_ocr: Optional[bool] = None, on_ocr_page: Optional[Callable[[], None]] = None) -> list:
    """Split one PDF into chunks, page by page, so every chunk knows its page.

    Pages without a text layer (scans) are read with OCR when it is available.

    Metadata on each chunk:
        source      path relative to the user's data folder, e.g. "Radar/Course/chap1.pdf"
        file_name   "chap1.pdf"
        course      top-level folder, e.g. "Radar"
        page        1-based page number in the PDF
        chunk_index position of the chunk within its page
        ocr         True if the text came from OCR (may contain recognition errors)
    """
    use_ocr = ocr_available() if use_ocr is None else use_ocr
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    course = _course_of(relative_path)
    cached = ocr_cache.load(file_path) if (use_ocr and ocr_cache) else {}
    new_ocr = {}
    chunks = []
    with pymupdf.open(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            from_ocr = False
            if use_ocr and needs_ocr(page):
                if page_number in cached:
                    text = cached[page_number]
                else:
                    text = new_ocr[page_number] = ocr_page(page)
                    if on_ocr_page:
                        on_ocr_page()
                from_ocr = True
            else:
                text = page.get_text()
            text = clean_text(text)
            if not text.strip():
                continue  # blank page, or scan with nothing readable
            for i, piece in enumerate(splitter.split_text(text)):
                chunks.append(Document(
                    page_content=piece,
                    metadata={
                        "source": relative_path.replace(os.sep, "/"),
                        "file_name": os.path.basename(file_path),
                        "course": course,
                        "page": page_number,
                        "chunk_index": i,
                        "ocr": from_ocr,
                    },
                ))
    if new_ocr and ocr_cache:
        ocr_cache.save(file_path, {**cached, **new_ocr})
    return chunks


def process_user_pdfs(username: str, progress: Optional[Callable[[str], None]] = None) -> list:
    """Chunk every PDF in the user's data folder, keeping file, course and page.

    progress(message) is called now and then (e.g. to update the setup status).
    """
    data_directory = data_path(username)
    if not os.path.exists(data_directory):
        print(f"Data directory not found for user '{username}'. Skipping processing.")
        return []

    pdfs = sorted(os.path.join(root, f) for root, _dirs, files in os.walk(data_directory)
                  for f in files if f.lower().endswith(".pdf"))
    ocr_cache = OcrCache(os.path.join(workspace_path(username), "ocr_cache"))
    use_ocr = ocr_available()
    if not use_ocr:
        print("OCR not available (pip install rapidocr_onnxruntime): scanned pages will be skipped.")

    print(f"Starting PDF processing for user '{username}' ({len(pdfs)} PDFs)...")
    all_chunks, ocr_pages = [], [0]

    def count_ocr_page():
        ocr_pages[0] += 1

    for n, file_path in enumerate(pdfs, start=1):
        relative = os.path.relpath(file_path, data_directory)
        if progress:
            note = f", {ocr_pages[0]} scanned pages read with OCR" if ocr_pages[0] else ""
            progress(f"Reading PDF {n}/{len(pdfs)}{note}: {relative}")
        try:
            all_chunks.extend(process_pdf(file_path, relative, ocr_cache, use_ocr, count_ocr_page))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(f"Processed {len(pdfs)} PDFs into {len(all_chunks)} chunks ({ocr_pages[0]} pages newly OCR'd).")
    return all_chunks
