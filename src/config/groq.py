from langchain_groq import ChatGroq
from .settings import GROQ_API_KEY, GROQ_MODEL


def get_groq_llm(model_name: str = GROQ_MODEL, temperature: float = 0):
    """Groq chat model (fast inference, generous free tier).

    Default model `openai/gpt-oss-20b`; override with GROQ_MODEL or the
    model_name arg. Run `GET https://api.groq.com/openai/v1/models` with your
    key to see which models your account can use. Free key:
    https://console.groq.com/keys
    """
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )
