"""Build and persist the FAISS vector store.

Run this once (or whenever the sources / embedding model change) so the app
can load the index instead of re-embedding on every start:

    python -m src.build_db                   # build if missing
    python -m src.build_db --rebuild         # force a fresh build
    python -m src.build_db --url URL ...      # override the source URLs
    python -m src.build_db --source PATH ...  # add extra files / directories
    python -m src.build_db --no-urls          # local files only

The knowledge base is: SOURCE_URLS + every .pdf / .docx / .txt / .md file
under KB_DIR (the `knowledge_base/` directory by default), plus anything
passed with --source.
"""
import argparse
import os

from .config import FAISS_INDEX_PATH, SOURCE_PATHS, SOURCE_URLS
from .retriever import index_documents, load_documents, save_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build the FAISS knowledge base.")
    parser.add_argument(
        "--path", default=FAISS_INDEX_PATH, help="where to save the index"
    )
    parser.add_argument(
        "--url", dest="urls", action="append", help="source URL (repeatable)"
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        help="extra file or directory to ingest (repeatable)",
    )
    parser.add_argument(
        "--no-urls", action="store_true", help="skip SOURCE_URLS, index local files only"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild even if an index already exists",
    )
    args = parser.parse_args()

    if os.path.isdir(args.path) and not args.rebuild:
        print(f"✅ Index already exists at '{args.path}'. Use --rebuild to overwrite.")
        return

    urls = [] if args.no_urls else (args.urls or SOURCE_URLS)
    paths = list(SOURCE_PATHS) + list(args.sources or [])

    print("📚 Building FAISS index from:")
    for u in urls:
        print(f"   url  {u}")
    for p in paths:
        print(f"   path {p}")

    docs = load_documents(urls, paths)
    print(f"→ loaded {len(docs)} document section(s); embedding & indexing...")

    vectorstore = index_documents(docs)
    save_vectorstore(vectorstore, args.path)
    print(f"✅ Saved index to '{args.path}'.")


if __name__ == "__main__":
    main()
