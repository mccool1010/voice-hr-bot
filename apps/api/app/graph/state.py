"""LangGraph state for one interview.

The state is a plain TypedDict so it serialises cleanly into the Postgres
checkpointer. It holds conversational state only — durable domain records live
in SQLAlchemy tables, which analytics queries directly.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from app.models.enums import Competency, Persona, Seniority, TurnKind


def append_only(existing: list[Any] | None, incoming: list[Any] | None) -> list[Any]:
    """Reducer: nodes return only the items they add, never the whole list."""
    return [*(existing or []), *(incoming or [])]


class TurnRecord(TypedDict, total=False):
    """One completed question/answer pair inside the graph."""

    index: int
    question: str
    answer: str
    competency: str
    kind: str
    # Populated by the evaluate node.
    overall: float
    structure: float
    specificity: float
    clarity: float
    depth: float
    notes: str
    model_score: float
    llm_score: float | None
    relevance: float | None
    features: dict[str, float]
    probe_focus: str | None


class InterviewState(TypedDict, total=False):
    # ─── Immutable configuration, set once at graph entry ──────────────────────
    interview_id: str
    role: str
    seniority: Seniority
    persona: Persona
    resume_context: str | None
    target_questions: int

    # ─── Plan, produced by the planner node ────────────────────────────────────
    plan: dict[str, Any]
    opening_remark: str

    # ─── Cursor ────────────────────────────────────────────────────────────────
    topic_index: int
    turn_index: int
    probe_depth: int
    current_question: str
    current_competency: Competency
    current_kind: TurnKind

    # ─── Accumulated history ───────────────────────────────────────────────────
    turns: Annotated[list[TurnRecord], append_only]

    # ─── Terminal output ───────────────────────────────────────────────────────
    finished: bool
    report: dict[str, Any] | None
    error: str | None

    # ─── Transient scratch space ───────────────────────────────────────────────
    # Written by one node and consumed by the next, then cleared. Declared here
    # because LangGraph silently discards any key absent from the state schema.
    _answer: str | None
    _word_timings: list[dict[str, Any]] | None
    _audio_duration_s: float | None
    _should_probe: bool


# How many consecutive follow-ups the interviewer may ask on a single topic
# before it must move on. Prevents the graph looping on one thread forever.
MAX_PROBE_DEPTH = 2


def initial_state(
    *,
    interview_id: str,
    role: str,
    seniority: Seniority,
    persona: Persona,
    target_questions: int,
    resume_context: str | None = None,
) -> InterviewState:
    return InterviewState(
        interview_id=interview_id,
        role=role,
        seniority=seniority,
        persona=persona,
        resume_context=resume_context,
        target_questions=target_questions,
        topic_index=0,
        turn_index=0,
        probe_depth=0,
        turns=[],
        finished=False,
        report=None,
        error=None,
    )
