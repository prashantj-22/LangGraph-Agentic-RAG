"""FastAPI wrapper around the Agentic RAG LangGraph app.

Run from the repo root:

    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs
"""
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src import service
from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    GROQ_MODEL,
    LLM_PROVIDER,
    SOURCE_URLS,
)

MODEL_NAME = GROQ_MODEL if LLM_PROVIDER == "groq" else "gpt-4o-mini"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        examples=["What are the key components of an AI agent?"],
    )


class Step(BaseModel):
    node: str
    type: str | None = None
    content: str | None = None
    query: str | None = None
    tool_calls: list[dict] | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    steps: list[Step]


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    embeddings: str
    index_path: str
    index_built: bool
    sources: list[str]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm up: load/build the FAISS index and compile the graph before serving.
    service.get_app()
    yield


app = FastAPI(
    title="Agentic RAG API",
    description="Self-correcting RAG over a LangGraph state machine.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": MODEL_NAME,
        "embeddings": EMBEDDING_MODEL,
        "index_path": FAISS_INDEX_PATH,
        "index_built": os.path.isdir(FAISS_INDEX_PATH),
        "sources": SOURCE_URLS,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Run the graph to completion and return the answer plus the node trace."""
    try:
        return service.answer_question(req.question)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Stream the reasoning trace as Server-Sent Events, then a final `answer`."""

    def event_gen():
        answer = ""
        try:
            for step, answer in service.iter_steps(req.question):
                yield f"event: step\ndata: {json.dumps(step)}\n\n"
            yield f"event: answer\ndata: {json.dumps({'answer': answer})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/rebuild-index")
def rebuild_index():
    """Rebuild the FAISS index from the configured source URLs."""
    try:
        service.rebuild_index()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "rebuilt", "index_path": FAISS_INDEX_PATH}
