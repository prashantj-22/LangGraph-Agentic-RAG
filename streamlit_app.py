"""Streamlit UI for the Agentic RAG system.

This is a thin client over the FastAPI service (`api.py`). Start the API first:

    uvicorn api:app --reload

then, in another shell:

    streamlit run streamlit_app.py

Point it at a different backend with the API_BASE_URL env var.
"""
import json
import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

# Node -> how to label it in the reasoning trace.
NODE_LABELS = {
    "agent": "🧠 Agent — decide whether to retrieve",
    "retrieve": "📚 Retrieve — search the vector store",
    "rewrite": "✏️ Rewrite — reformulate the query and retry",
    "generate": "✨ Generate — answer from the retrieved context",
}


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class APIError(Exception):
    pass


def api_get(path: str, timeout: float = 10.0) -> dict:
    try:
        r = httpx.get(f"{API_BASE_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise APIError(str(exc)) from exc


def api_post(path: str, body: dict | None = None, timeout: float = 300.0) -> dict:
    try:
        r = httpx.post(f"{API_BASE_URL}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise APIError(str(exc)) from exc


def stream_ask(question: str):
    """Yield (event, payload) tuples from the /ask/stream SSE endpoint."""
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE_URL}/ask/stream",
            json={"question": question},
            timeout=300.0,
        ) as r:
            r.raise_for_status()
            event = "message"
            for line in r.iter_lines():
                if not line:
                    event = "message"
                    continue
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    yield event, json.loads(line.split(":", 1)[1].strip())
    except httpx.HTTPError as exc:
        raise APIError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_step(step: dict):
    node = step.get("node", "?")
    label = NODE_LABELS.get(node, f"🔹 {node}")
    with st.expander(label, expanded=False):
        kind = step.get("type")
        if kind == "tool_call":
            for tc in step.get("tool_calls", []):
                st.markdown(f"**Tool call:** `{tc['name']}`")
                st.json(tc["args"])
        elif kind == "documents":
            st.markdown("**Retrieved documents:**")
            st.text(step.get("content", "")[:2000])
        elif kind == "rewrite":
            st.markdown(f"**Rewritten query:** {step.get('query', '')}")
        elif kind == "message":
            st.markdown(step.get("content") or "_(no content)_")
        else:
            st.json(step)


def run_query(question: str) -> str:
    final_answer = ""
    steps_box = st.container()
    for event, payload in stream_ask(question):
        if event == "step":
            with steps_box:
                render_step(payload)
        elif event == "answer":
            final_answer = payload.get("answer", "")
        elif event == "error":
            st.error(f"API error: {payload.get('detail', 'unknown')}")
    return final_answer or "_No answer produced._"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Backend")
    st.caption(f"API: `{API_BASE_URL}`")

    try:
        health = api_get("/health")
    except APIError as exc:
        st.error(
            f"Can't reach the API at `{API_BASE_URL}`.\n\n"
            f"Start it with `uvicorn api:app --reload`.\n\n`{exc}`"
        )
        st.stop()

    st.success("Connected")
    st.markdown(
        f"""
| Setting | Value |
| --- | --- |
| LLM provider | `{health.get('provider')}` |
| Chat model | `{health.get('model')}` |
| Embeddings | `{health.get('embeddings')}` |
| Index built | `{health.get('index_built')}` |
"""
    )

    st.subheader("📄 Knowledge base")
    for url in health.get("sources", []):
        st.markdown(f"- [{url}]({url})")

    if st.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding index via API…"):
            try:
                api_post("/rebuild-index", timeout=600.0)
                st.success("Index rebuilt.")
            except APIError as exc:
                st.error(f"Rebuild failed: {exc}")

    st.caption(
        "Each question is answered independently — the graph has no "
        "conversation memory (no checkpointer)."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("🤖 Agentic RAG with LangGraph")
st.caption(
    "The agent decides whether to retrieve, grades the retrieved documents, "
    "and rewrites the query to try again when they are weak. "
    "This UI talks to the FastAPI backend over HTTP."
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer)

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

sample = "What are the key components of an AI agent?"
question = st.chat_input(f'Ask a question (e.g. "{sample}")')

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Running the graph via the API…", expanded=True):
            try:
                answer = run_query(question)
            except APIError as exc:
                st.error(f"Request failed: {exc}")
                answer = "_Request failed._"
        st.markdown("### Answer")
        st.markdown(answer)

    st.session_state.history.append((question, answer))
