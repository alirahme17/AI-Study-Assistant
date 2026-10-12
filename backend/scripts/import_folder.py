"""Import a folder of PDFs straight into the database (admin / development tool).

Each top-level sub-folder becomes a course, like a folder upload in the app:

    python scripts/import_folder.py --user you@example.com  "C:/path/to/data"
    python scripts/import_folder.py --user dev-alice ./data --no-index   # only upload, index later

Needs the same environment as the backend (DATABASE_URL, storage settings, ENCRYPTION_KEY).
Indexing runs here, in this process, with the user's saved Gemini key (or EMBEDDINGS=fake).
Most people should use the app's "Upload folder" button instead.
"""
import argparse
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "eval"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="account e-mail, user id, or dev-mode test user name")
    parser.add_argument("folder")
    parser.add_argument("--no-index", action="store_true", help="only upload; the app indexes them later")
    args = parser.parse_args(argv)

    import db
    import documents
    import indexer
    from run_eval import resolve_user

    with db.connect() as conn:
        db.apply_schema(conn)
        user_id = resolve_user(conn, args.user)
    root = os.path.abspath(args.folder)
    pdfs = sorted(os.path.join(d, f) for d, _, files in os.walk(root) for f in files if f.lower().endswith(".pdf"))
    print(f"{len(pdfs)} PDFs found under {root}")
    for n, path in enumerate(pdfs, 1):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        course, _, name = rel.partition("/")
        if not name:
            course, name = "General", rel
        with open(path, "rb") as f:
            data = f.read()
        with db.connect() as conn:
            source = documents.add_document(conn, user_id, documents.clean_course(course),
                                            documents.clean_name(name), data)["source"]
        print(f"[{n}/{len(pdfs)}] {source}")

    if not args.no_index:
        print("Indexing...")
        with indexer._lock:
            indexer._running.add(user_id)
        indexer.run_worker(user_id)
        with db.connect() as conn:
            docs = documents.list_documents(conn, user_id)
        status = documents.overall_status(docs)
        print(status["message"])
        for d in docs:
            if d["status"] != "ready":
                print(f"  {d['status']}: {d['source']} - {d['message']}")


if __name__ == "__main__":
    main()
