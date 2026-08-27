"""Streamlit UI for the Agentic RAG system.

Run from the repo root:

    streamlit run streamlit_app.py
"""
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    GROQ_MODEL,
    LLM_PROVIDER,
    SOURCE_URLS,
)
from src.retriever import get_vectorstore, get_retriever_tool
from src.agents.graph import build_graph

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

MODEL_NAME = GROQ_MODEL if LLM_PROVIDER == "groq" else "gpt-4o-mini"

# Node -> how to label it in the reasoning trace.
NODE_LABELS = {
    "agent": "🧠 Agent — decide whether to retrieve",
    "retrieve": "📚 Retrieve — search the vector store",
    "rewrite": "✏️ Rewrite — reformulate the query and retry",
    "generate": "✨ Generate — answer from the retrieved context",
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading / building the FAISS index…")
def load_app(_rebuild_token: int = 0):
    """Build the retriever tool + compiled LangGraph app.

    `_rebuild_token` is bumped to bust the cache when the user rebuilds.
    """
    vectorstore = get_vectorstore(rebuild=_rebuild_token > 0)
    tools = [get_retriever_tool(vectorstore)]
    return build_graph(tools)


def render_step(node: str, value: dict):
    """Render one node's output from the graph stream."""
    label = NODE_LABELS.get(node, f"🔹 {node}")
    msgs = value.get("messages", []) if isinstance(value, dict) else []
    last = msgs[-1] if msgs else None

    with st.expander(label, expanded=False):
        if isinstance(last, AIMessage) and last.tool_calls:
            for tc in last.tool_calls:
                st.markdown(f"**Tool call:** `{tc['name']}`")
                st.json(tc["args"])
        elif isinstance(last, ToolMessage):
            st.markdown("**Retrieved documents:**")
            st.text((last.content or "")[:2000])
        elif isinstance(last, HumanMessage):
            st.markdown(f"**Rewritten query:** {last.content}")
        elif isinstance(last, AIMessage):
            st.markdown(last.content or "_(no content)_")
        else:
            st.write(value)


def run_query(app, question: str) -> str:
    """Stream the graph, render each step, return the final answer text."""
    inputs = {"messages": [HumanMessage(content=question)]}
    final_answer = ""

    steps_box = st.container()
    for output in app.stream(inputs):
        for node, value in output.items():
            with steps_box:
                render_step(node, value)
            msgs = value.get("messages", []) if isinstance(value, dict) else []
            if msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].content:
                final_answer = msgs[-1].content

    return final_answer or "_No answer produced._"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(
        f"""
| Setting | Value |
| --- | --- |
| LLM provider | `{LLM_PROVIDER}` |
| Chat model | `{MODEL_NAME}` |
| Embeddings | `{EMBEDDING_MODEL}` |
| Index path | `{FAISS_INDEX_PATH}` |
| Index built | `{os.path.isdir(FAISS_INDEX_PATH)}` |
"""
    )

    st.subheader("📄 Knowledge base")
    for url in SOURCE_URLS:
        st.markdown(f"- [{url}]({url})")

    if "rebuild_token" not in st.session_state:
        st.session_state.rebuild_token = 0

    if st.button("🔄 Rebuild index", use_container_width=True):
        st.session_state.rebuild_token += 1
        load_app.clear()
        get_vectorstore(rebuild=True)
        st.success("Index rebuilt.")
        st.rerun()

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
    "and rewrites the query to try again when they are weak."
)

app = load_app(st.session_state.get("rebuild_token", 0))

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer)

# Replay prior turns.
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
        with st.status("Running the graph…", expanded=True):
            answer = run_query(app, question)
        st.markdown("### Answer")
        st.markdown(answer)

    st.session_state.history.append((question, answer))
