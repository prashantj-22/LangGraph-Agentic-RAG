import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Groq — fast inference with a generous free tier.
# Get a key at https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Which provider get_llm() uses by default: "openai" or "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Open-source embeddings (sentence-transformers, runs locally, no API key).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Where the persisted FAISS index lives, so it is built once and reused.
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# Documents ingested into the knowledge base.
SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

if LLM_PROVIDER == "groq":
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set in .env file")
elif not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")
