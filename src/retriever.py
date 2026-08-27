import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

from .config import get_embeddings, FAISS_INDEX_PATH, SOURCE_URLS


def build_vectorstore(urls: list[str] = SOURCE_URLS):
    """Load, split and embed the source URLs into a fresh FAISS store."""
    loader = WebBaseLoader(urls)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    return FAISS.from_documents(documents=splits, embedding=get_embeddings())


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
    path: str = FAISS_INDEX_PATH,
    rebuild: bool = False,
):
    """Return the FAISS store, building and persisting it only when needed.

    Loads the index from `path` if it already exists; otherwise builds it from
    `urls` and saves it there for next time. Pass `rebuild=True` to force a
    fresh build.
    """
    if not rebuild and os.path.isdir(path):
        return load_vectorstore(path)

    vectorstore = build_vectorstore(urls)
    save_vectorstore(vectorstore, path)
    return vectorstore


# Backwards-compatible alias.
def ingest_documents(urls: list[str] = SOURCE_URLS):
    return build_vectorstore(urls)


def get_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and retrieve relevant documents from the knowledge base. Use this when you need external information to answer the question.",
    )

    return retriever_tool
