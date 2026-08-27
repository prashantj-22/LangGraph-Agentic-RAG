from langchain_openai import ChatOpenAI
from .settings import OPENAI_API_KEY, GROQ_MODEL, LLM_PROVIDER
from .groq import get_groq_llm


def get_openai_llm(model_name: str = "gpt-4o-mini", temperature: float = 0):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )


def get_llm(model_name: str | None = None, temperature: float = 0, provider: str | None = None):
    """Return the default chat model.

    Provider is chosen by the `provider` arg, else the LLM_PROVIDER env var
    ("openai" or "groq"), defaulting to OpenAI.
    """
    provider = (provider or LLM_PROVIDER).lower()
    if provider == "groq":
        return get_groq_llm(model_name or GROQ_MODEL, temperature)
    return get_openai_llm(model_name or "gpt-4o-mini", temperature)
