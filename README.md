# Agentic RAG System with LangGraph

A production-ready Retrieval-Augmented Generation (RAG) system that automatically corrects retrieval errors using LangGraph for orchestration and FAISS for vector storage.

## Features

- **Self-Correcting Retrieval**: Automatically detects and fixes poor retrieval quality
- **Transparent Decision-Making**: Clear state machine with explicit routing decisions
- **Module Decoupling**: Easy to swap components (Redis → Pinecone, OpenAI → Anthropic)
- **Document Grading**: LLM-based relevance scoring before generation
- **Query Rewriting**: Transforms unclear queries into search-friendly formats

## Architecture

The system consists of 6 key components:

1. **Configuration Layer**: Manages environment variables and API clients
2. **Retrieval Module**: Handles document ingestion, vectorization, and storage
3. **Agent Node**: Decision-making entry point
4. **Grade Edge**: Quality checkpoint for search results
5. **Rewrite Node**: Query transformation for better results
6. **Generation Node**: Final answer production

## Project Structure

```
streamlit_app.py         # Streamlit chat UI
src/
├── config/
│   ├── settings.py      # Environment variables
│   ├── openai.py        # OpenAI client + get_llm() provider router
│   ├── groq.py          # Groq client
│   └── embeddings.py    # Local sentence-transformers embeddings
├── retriever.py         # Document ingestion, FAISS build/save/load, retriever tool
├── build_db.py          # One-off CLI to build & persist the FAISS index
├── agents/
│   ├── nodes.py         # Agent, rewrite, and generate functions
│   ├── edges.py         # Routing + document grading logic
│   └── graph.py         # LangGraph state machine
└── main.py              # CLI entry point
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- OpenAI API key

### 1. Create Virtual Environment with uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e .
```

### 2. Configure Environment

Your `.env` file should already contain:
```
OPENAI_API_KEY=your_api_key_here
```

#### Optional: use Groq for the LLM

Groq offers fast inference with a generous free tier. Get a key at
https://console.groq.com/keys and add to `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
# optional override (default: openai/gpt-oss-20b)
# check what your key can use: GET https://api.groq.com/openai/v1/models
GROQ_MODEL=openai/gpt-oss-20b
```
Embeddings use an open-source `sentence-transformers` model that runs locally
(no API key), so with `LLM_PROVIDER=groq` you don't need `OPENAI_API_KEY` at
all. To use Groq for a single call: `get_llm(provider="groq")`.

#### Embeddings

Embeddings default to `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim).
Override with `EMBEDDING_MODEL=...` in `.env`. The model downloads from
HuggingFace on first use.

### 3. Build the Vector Store (once)

The FAISS index is built once and persisted to `faiss_index/`, so it is not
re-embedded on every run.

```bash
python -m src.build_db              # build if missing
python -m src.build_db --rebuild    # force a fresh build (sources/model changed)
```

Configure with `FAISS_INDEX_PATH` in `.env` (default `faiss_index`). Source URLs
live in `SOURCE_URLS` in [src/config/settings.py](src/config/settings.py).

### 4. Run the System

**CLI:**

```bash
python -m src.main
```

**Streamlit UI:**

```bash
streamlit run streamlit_app.py
```

The UI gives you a chat box, an expandable reasoning trace for every graph node
(agent / retrieve / rewrite / generate), a config panel, and a "Rebuild index"
button in the sidebar.

Both entry points load the persisted index automatically, building it on first
run if `faiss_index/` doesn't exist yet.

## How It Works

### Workflow

1. **Agent receives question** → Decides to retrieve or answer directly
2. **Retrieval** → Fetches relevant documents from FAISS vector store
3. **Grading** → LLM strictly evaluates document relevance with structured output
4. **Decision point**:
   - ✅ Relevant → Generate answer
   - ❌ Not relevant → Rewrite query and retry
5. **Generation** → Produces final answer based on verified context

### State Machine Flow

```
Start → Agent → Retrieve → Grade Documents
                              ├─ Relevant → Generate → End
                              └─ Not Relevant → Rewrite → Agent
```

## Example Output

```
❓ Question: What are the key components of an AI agent?

🔄 Processing...

📍 Node: agent
🔧 Tool Call: retrieve_documents

📍 Node: retrieve
💬 Output: Retrieved documents...

📍 Node: generate
💬 Output: Based on the documents...

✨ FINAL ANSWER:
The key components of an AI agent include:
1. Planning and reasoning capabilities
2. Memory systems for context retention
3. Tool use and execution
4. Reflection and self-improvement mechanisms
```

## Customization

### Change Vector Store

Modify `src/retriever.py`:
```python
# Replace FAISS with Pinecone, Weaviate, etc.
from langchain_pinecone import PineconeVectorStore
```

### Change LLM Provider

Modify `src/config/openai.py`:
```python
from langchain_anthropic import ChatAnthropic

def get_llm():
    return ChatAnthropic(model="claude-3-sonnet-20240229")
```

### Adjust Chunk Size

Modify `src/retriever.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks
    chunk_overlap=100
)
```

## Key Advantages

- **Self-Correction**: Detects and fixes poor retrieval quality
- **Transparent**: Every decision is logged and traceable
- **Modular**: Easy to replace components without breaking the system
- **Production-Ready**: Handles edge cases that simpler RAG systems miss

## Dependencies

- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community components
- `langgraph` - State machine orchestration
- `faiss-cpu` - Local vector store
- `beautifulsoup4` - Web scraping
- `python-dotenv` - Environment management
- `tiktoken` - Token counting

## Troubleshooting

### OpenAI API Error
```bash
# Verify API key is set
echo $OPENAI_API_KEY
```

### Import Errors
```bash
# Reinstall dependencies
uv pip install -e . --force-reinstall
```

## References

Based on the article about building Agentic RAG systems with self-correcting retrieval capabilities using LangGraph and vector stores.

For a step-by-step build walkthrough and a node-by-node explanation of the
LangGraph workflow (with interview talking points), see
[docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md).

## Contributing

Feel free to submit issues and enhancement requests!
