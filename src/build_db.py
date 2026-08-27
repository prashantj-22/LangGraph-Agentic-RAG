"""Build and persist the FAISS vector store.

Run this once (or whenever the sources / embedding model change) so the app
can load the index instead of re-embedding on every start:

    python -m src.build_db                # build if missing
    python -m src.build_db --rebuild      # force a fresh build
    python -m src.build_db --url URL ...  # override the source URLs
"""
import argparse
import os

from .config import FAISS_INDEX_PATH, SOURCE_URLS
from .retriever import build_vectorstore, save_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build the FAISS knowledge base.")
    parser.add_argument(
        "--path", default=FAISS_INDEX_PATH, help="where to save the index"
    )
    parser.add_argument(
        "--url", dest="urls", action="append", help="source URL (repeatable)"
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

    urls = args.urls or SOURCE_URLS
    print(f"📚 Building FAISS index from {len(urls)} source(s)...")
    for u in urls:
        print(f"   - {u}")

    vectorstore = build_vectorstore(urls)
    save_vectorstore(vectorstore, args.path)
    print(f"✅ Saved index to '{args.path}'.")


if __name__ == "__main__":
    main()
