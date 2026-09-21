"""Graph assembly and checkpointing.

The compiled graph is a process-wide singleton — building it is cheap but not
free, and it holds a connection pool for the Postgres checkpointer.

Checkpointing is what makes an interview resumable: every `interrupt` writes the
full state, so a dropped WebSocket, a browser refresh or an API restart all
resume exactly where the candidate left off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph.nodes import (
    advance,
    ask_question,
    await_answer,
    compose_report,
    evaluate_answer,
    plan_interview,
    route_after_advance,
    route_after_plan,
)
from app.graph.state import InterviewState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

log = structlog.get_logger(__name__)

_graph: Any = None
_checkpointer: Any = None
_pool_cm: Any = None


def build_graph(checkpointer: Any) -> CompiledStateGraph:
    """Wire the interview state machine."""
    builder = StateGraph(InterviewState)

    builder.add_node("plan", plan_interview)
    builder.add_node("ask", ask_question)
    builder.add_node("await_answer", await_answer)
    builder.add_node("evaluate", evaluate_answer)
    builder.add_node("advance", advance)
    builder.add_node("report", compose_report)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_after_plan, {"ask": "ask", "report": "report"})
    builder.add_edge("ask", "await_answer")
    builder.add_edge("await_answer", "evaluate")
    builder.add_edge("evaluate", "advance")
    builder.add_conditional_edges(
        "advance", route_after_advance, {"ask": "ask", "report": "report"}
    )
    builder.add_edge("report", END)

    return builder.compile(checkpointer=checkpointer)


async def init_graph() -> None:
    """Called from the application lifespan. Sets up durable checkpointing."""
    global _graph, _checkpointer, _pool_cm

    if _graph is not None:
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        _pool_cm = AsyncPostgresSaver.from_conn_string(settings.checkpoint_database_url)
        _checkpointer = await _pool_cm.__aenter__()
        await _checkpointer.setup()
        log.info("graph.checkpointer", backend="postgres")
    except Exception as exc:
        # An in-memory saver keeps the app working (interviews just do not
        # survive a restart), which matters for local runs without Postgres.
        log.warning("graph.checkpointer_fallback", backend="memory", error=str(exc))
        _checkpointer = InMemorySaver()
        _pool_cm = None

    _graph = build_graph(_checkpointer)
    log.info("graph.compiled")


async def shutdown_graph() -> None:
    global _graph, _checkpointer, _pool_cm
    if _pool_cm is not None:
        try:
            await _pool_cm.__aexit__(None, None, None)
        except Exception as exc:  # pragma: no cover
            log.warning("graph.checkpointer_close_failed", error=str(exc))
    _graph = None
    _checkpointer = None
    _pool_cm = None


def get_graph() -> CompiledStateGraph:
    """The compiled graph. Falls back to an in-memory saver if not initialised."""
    global _graph, _checkpointer
    if _graph is None:
        log.warning("graph.lazy_init", reason="init_graph was not awaited")
        _checkpointer = InMemorySaver()
        _graph = build_graph(_checkpointer)
    return _graph


def set_graph(graph: Any) -> None:
    """Test hook."""
    global _graph
    _graph = graph


def thread_config(thread_id: str) -> RunnableConfig:
    """Checkpointer addressing for one interview."""
    return {"configurable": {"thread_id": thread_id}}
