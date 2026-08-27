import os
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

from .config import get_embeddings, FAISS_INDEX_PATH, SOURCE_PATHS, SOURCE_URLS

# File types we can pull into the knowledge base, and how to load each.
LOADER_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda p: TextLoader(p, autodetect_encoding=True),
    ".md": lambda p: TextLoader(p, autodetect_encoding=True),
}
SUPPORTED_SUFFIXES = tuple(LOADER_BY_SUFFIX)


def _iter_files(path: Path):
    """Yield supported files under `path` (a single file or a directory tree)."""
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        return
    for f in candidates:
        if f.suffix.lower() in LOADER_BY_SUFFIX:
            yield f
        elif f.suffix.lower() == ".doc":
            print(f"⚠️  Skipping {f.name}: legacy .doc is unsupported, convert it to .docx")


def load_file_documents(paths: list[str] = SOURCE_PATHS) -> list[Document]:
    """Load every supported document found under the given files / directories."""
    docs: list[Document] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        for f in _iter_files(p):
            loader = LOADER_BY_SUFFIX[f.suffix.lower()](str(f))
            loaded = loader.load()
            for d in loaded:
                d.metadata.setdefault("source", str(f))
            docs.extend(loaded)
    return docs


def load_documents(
    urls: list[str] = SOURCE_URLS,
    paths: list[str] = SOURCE_PATHS,
) -> list[Document]:
    """Load knowledge-base documents from web URLs and local files."""
    docs: list[Document] = []
    if urls:
        docs.extend(WebBaseLoader(list(urls)).load())
    docs.extend(load_file_documents(paths))
    return docs


def index_documents(docs: list[Document]):
    """Split the given documents and embed them into a fresh FAISS store."""
    if not docs:
        raise ValueError(
            "No documents to index. Add URLs to SOURCE_URLS or drop files into "
            "the knowledge base directory (KB_DIR)."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    return FAISS.from_documents(documents=splits, embedding=get_embeddings())


def build_vectorstore(
    urls: list[str] = SOURCE_URLS,
    paths: list[str] = SOURCE_PATHS,
):
    """Load, split and embed the sources into a fresh FAISS store."""
    return index_documents(load_documents(urls, paths))


def save_vectorstore(vectorstore, path: str = FAISS_INDEX_PATH):
    vectorstore.save_local(path)
    return path


def load_vectorstore(path: str = FAISS_INDEX_PATH):
    return FAISS.load_local(
        path,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_vectorstore(
    urls: list[str] = SOURCE_URLS,
    paths: list[str] = SOURCE_PATHS,
    path: str = FAISS_INDEX_PATH,
    rebuild: bool = False,
):
    """Return the FAISS store, building and persisting it only when needed.

    Loads the index from `path` if it already exists; otherwise builds it from
    `urls` + `paths` and saves it there for next time. Pass `rebuild=True` to
    force a fresh build.
    """
    if not rebuild and os.path.isdir(path):
        return load_vectorstore(path)

    vectorstore = build_vectorstore(urls, paths)
    save_vectorstore(vectorstore, path)
    return vectorstore


# Backwards-compatible alias.
def ingest_documents(
    urls: list[str] = SOURCE_URLS,
    paths: list[str] = SOURCE_PATHS,
):
    return build_vectorstore(urls, paths)


def get_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and retrieve relevant documents from the knowledge base. Use this when you need external information to answer the question.",
    )

    return retriever_tool
