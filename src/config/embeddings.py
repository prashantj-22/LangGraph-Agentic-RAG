from langchain_huggingface import HuggingFaceEmbeddings
from .settings import EMBEDDING_MODEL


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """Open-source sentence-transformers embeddings (runs locally, no API key).

    Default `all-MiniLM-L6-v2` is a small, fast 384-dim model. Override with
    the EMBEDDING_MODEL env var or the model_name arg.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
