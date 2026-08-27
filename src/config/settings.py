import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = two levels up from this file (src/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the repo-root .env explicitly so it is found regardless of CWD.
load_dotenv(PROJECT_ROOT / ".env")

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
# A relative value is anchored to the repo root, not the current directory,
# so the app works no matter where it is launched from.
_faiss_path = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index")).expanduser()
if not _faiss_path.is_absolute():
    _faiss_path = PROJECT_ROOT / _faiss_path
FAISS_INDEX_PATH = str(_faiss_path)

# Web pages ingested into the knowledge base.
SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

# Local documents ingested into the knowledge base: drop .pdf / .docx / .txt /
# .md files into this directory (scanned recursively). Relative -> repo root.
_kb_dir = Path(os.getenv("KB_DIR", "knowledge_base")).expanduser()
if not _kb_dir.is_absolute():
    _kb_dir = PROJECT_ROOT / _kb_dir
KB_DIR = str(_kb_dir)
SOURCE_PATHS = [KB_DIR]

if LLM_PROVIDER == "groq":
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set in .env file")
elif not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")
