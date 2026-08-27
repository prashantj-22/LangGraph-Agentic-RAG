"""Shared service layer: run the LangGraph app and shape its output.

Used by both the FastAPI app (`api.py`) and available for reuse elsewhere.
"""
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .agents.graph import build_graph
from .retriever import get_retriever_tool, get_vectorstore


@lru_cache(maxsize=1)
def get_app():
    """Build (once) the compiled LangGraph app with the retriever tool bound."""
    vectorstore = get_vectorstore()
    return build_graph([get_retriever_tool(vectorstore)])


def summarize_step(node: str, value: dict) -> dict:
    """Turn one raw stream item ({node: {"messages": [...]}}) into a JSON dict."""
    msgs = value.get("messages", []) if isinstance(value, dict) else []
    last = msgs[-1] if msgs else None
    step: dict = {"node": node}

    if isinstance(last, AIMessage) and last.tool_calls:
        step["type"] = "tool_call"
        step["tool_calls"] = [
            {"name": tc["name"], "args": tc["args"]} for tc in last.tool_calls
        ]
    elif isinstance(last, ToolMessage):
        step["type"] = "documents"
        step["content"] = (last.content or "")[:4000]
    elif isinstance(last, HumanMessage):
        step["type"] = "rewrite"
        step["query"] = last.content
    elif isinstance(last, AIMessage):
        step["type"] = "message"
        step["content"] = last.content or ""
    else:
        step["type"] = "unknown"

    return step


def iter_steps(question: str):
    """Yield (step_dict, running_answer) as the graph executes."""
    app = get_app()
    inputs = {"messages": [HumanMessage(content=question)]}
    answer = ""

    for output in app.stream(inputs):
        for node, value in output.items():
            step = summarize_step(node, value)
            msgs = value.get("messages", []) if isinstance(value, dict) else []
            if msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].content:
                answer = msgs[-1].content
            yield step, answer


def answer_question(question: str) -> dict:
    """Run the full graph and return {question, answer, steps}."""
    steps: list[dict] = []
    answer = ""
    for step, answer in iter_steps(question):
        steps.append(step)
    return {
        "question": question,
        "answer": answer or "No answer produced.",
        "steps": steps,
    }


def rebuild_index() -> None:
    """Force a fresh FAISS build and drop the cached graph."""
    get_vectorstore(rebuild=True)
    get_app.cache_clear()
