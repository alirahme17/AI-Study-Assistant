import os
import shutil
from typing import Callable, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from process import process_user_pdfs
from workspace import new_db_path, workspace_path

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_user_database(username: str, progress: Optional[Callable[[str], None]] = None) -> str:
    """Build a fresh ChromaDB for the user and return its folder name.

    Raises if there is nothing to index, so the caller can report an error
    instead of waiting forever for a database that will never appear.
    """
    chunks = process_user_pdfs(username, progress)
    if not chunks:
        raise RuntimeError(
            "No readable text was found in your PDFs. Check that the Drive folder "
            "contains PDFs (scanned PDFs without a text layer are not supported yet)."
        )

    if progress:
        progress(f"Building the search index ({len(chunks)} chunks)...")
    persist_directory = new_db_path(username)
    print(f"Building vector database for user '{username}' ({len(chunks)} chunks)...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            persist_directory=persist_directory,
        )
    except Exception:
        shutil.rmtree(persist_directory, ignore_errors=True)
        raise
    print(f"Vector database for user '{username}' built successfully!")
    return os.path.basename(persist_directory)


def remove_old_databases(username: str, keep_dir: str) -> None:
    """Best-effort cleanup of previous builds (a file still open on Windows is skipped)."""
    root = workspace_path(username)
    for name in os.listdir(root):
        if name.startswith("chroma_db") and name != keep_dir and os.path.isdir(os.path.join(root, name)):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
