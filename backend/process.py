import os

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from workspace import data_path

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _course_of(relative_path: str) -> str:
    parts = relative_path.split(os.sep)
    return parts[0] if len(parts) > 1 else "Unknown"


def process_pdf(file_path: str, relative_path: str) -> list:
    """Split one PDF into chunks, page by page, so every chunk knows its page.

    Metadata on each chunk:
        source      path relative to the user's data folder, e.g. "Radar/Course/chap1.pdf"
        file_name   "chap1.pdf"
        course      top-level folder, e.g. "Radar"
        page        1-based page number in the PDF
        chunk_index position of the chunk within its page
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    course = _course_of(relative_path)
    chunks = []
    with pymupdf.open(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue  # blank or scanned page with no text layer
            for i, piece in enumerate(splitter.split_text(text)):
                chunks.append(Document(
                    page_content=piece,
                    metadata={
                        "source": relative_path.replace(os.sep, "/"),
                        "file_name": os.path.basename(file_path),
                        "course": course,
                        "page": page_number,
                        "chunk_index": i,
                    },
                ))
    return chunks


def process_user_pdfs(username: str) -> list:
    """Chunk every PDF in the user's data folder, keeping file, course and page."""
    data_directory = data_path(username)
    if not os.path.exists(data_directory):
        print(f"Data directory not found for user '{username}'. Skipping processing.")
        return []

    print(f"Starting PDF processing for user '{username}'...")
    all_chunks = []
    for root, _dirs, files in os.walk(data_directory):
        for file in sorted(files):
            if not file.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(root, file)
            try:
                all_chunks.extend(process_pdf(file_path, os.path.relpath(file_path, data_directory)))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return all_chunks
