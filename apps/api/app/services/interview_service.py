"""Bridge between the LangGraph state machine and the database.

The graph owns conversational state; SQLAlchemy owns the queryable record.
This module is the only place that knows about both, so neither side has to
import the other.

Every graph invocation ends in one of two places — suspended on an `interrupt`
waiting for an answer, or finished with a report — and `_advance` normalises
that into a single return type the routers can act on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.aggregate import build_report_delivery
from app.graph.builder import get_graph, thread_config
from app.graph.state import initial_state
from app.llm.factory import get_provider
from app.models.enums import Competency, InterviewStatus, TurnKind
from app.models.interview import Interview, Turn
from app.models.resume import Resume
from app.models.score import InterviewReport, TurnScore
from app.models.user import User
from app.scoring.model import SCORER_VERSION

log = structlog.get_logger(__name__)

MAX_RESUME_CONTEXT_CHARS = 6000


class InterviewError(RuntimeError):
    """Domain error the router turns into a 4xx."""


@dataclass
class GraphStep:
    """Normalised result of advancing the graph."""

    question: str | None = None
    competency: Competency = Competency.communication
    kind: TurnKind = TurnKind.main
    turn_index: int = 0
    finished: bool = False
    report: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


def _extract_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the pending interrupt payload out of an invoke result.

    LangGraph surfaces interrupts under `__interrupt__` as a tuple of Interrupt
    objects. The shape has shifted across releases, so this tolerates both an
    object with `.value` and a bare dict.
    """
    raw = result.get("__interrupt__")
    if not raw:
        return None

    first = raw[0] if isinstance(raw, list | tuple) else raw
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else None


async def _advance(config: RunnableConfig, payload: Any) -> GraphStep:
    """Run the graph until it suspends or completes."""
    graph = get_graph()
    result: dict[str, Any] = await graph.ainvoke(payload, config=config)

    if pending := _extract_interrupt(result):
        return GraphStep(
            question=pending.get("question", ""),
            competency=Competency(pending.get("competency", Competency.communication)),
            kind=TurnKind(pending.get("kind", TurnKind.main)),
            turn_index=int(pending.get("turn_index", 0)),
            finished=False,
            state=result,
        )

    return GraphStep(
        finished=True,
        report=result.get("report"),
        turn_index=int(result.get("turn_index", 0)),
        state=result,
    )


async def _resume_context(
    db: AsyncSession, resume_id: uuid.UUID | None, user_id: uuid.UUID
) -> str | None:
    if resume_id is None:
        return None
    resume = await db.scalar(
        select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id)
    )
    if resume is None:
        raise InterviewError("Resume not found.")

    facts = resume.extracted or {}
    header = ""
    if facts:
        skills = ", ".join(facts.get("skills", [])[:20])
        projects = "; ".join(facts.get("projects", [])[:8])
        header = (
            f"Headline: {facts.get('headline', '')}\n"
            f"Years of experience: {facts.get('years_experience', 'unknown')}\n"
            f"Skills: {skills}\n"
            f"Projects: {projects}\n\n"
        )
    return (header + resume.raw_text)[:MAX_RESUME_CONTEXT_CHARS]


# ─── Public API ───────────────────────────────────────────────────────────────


async def create_interview(
    db: AsyncSession,
    user: User,
    *,
    role: str,
    seniority: Any,
    persona: Any,
    target_questions: int,
    resume_id: uuid.UUID | None = None,
) -> tuple[Interview, GraphStep]:
    """Create the record, plan the interview, and return the first question."""
    provider = get_provider()
    thread_id = uuid.uuid4().hex

    interview = Interview(
        user_id=user.id,
        resume_id=resume_id,
        role=role.strip(),
        seniority=seniority,
        persona=persona,
        status=InterviewStatus.in_progress,
        thread_id=thread_id,
        target_questions=target_questions,
        llm_provider=provider.name,
        llm_model=provider.model,
        started_at=datetime.now(UTC),
    )
    db.add(interview)
    await db.flush()

    context = await _resume_context(db, resume_id, user.id)

    step = await _advance(
        thread_config(thread_id),
        initial_state(
            interview_id=str(interview.id),
            role=interview.role,
            seniority=seniority,
            persona=persona,
            target_questions=target_questions,
            resume_context=context,
        ),
    )

    if step.state and step.state.get("plan"):
        interview.plan = step.state["plan"]

    if step.finished:
        # Planning failed — do not leave a half-created interview in progress.
        interview.status = InterviewStatus.abandoned
        interview.completed_at = datetime.now(UTC)
        await db.commit()
        raise InterviewError(
            "The interview could not be started because the language model was "
            "unavailable. Check the provider configuration and try again."
        )

    db.add(
        Turn(
            interview_id=interview.id,
            index=step.turn_index,
            kind=step.kind,
            competency=step.competency,
            question=step.question or "",
        )
    )
    await db.commit()
    await db.refresh(interview)

    log.info("interview.created", interview_id=str(interview.id), role=interview.role)
    return interview, step


async def submit_answer(
    db: AsyncSession,
    interview: Interview,
    *,
    answer: str,
    word_timings: list[dict[str, Any]] | None = None,
    audio_duration_s: float | None = None,
    transcription_confidence: float | None = None,
) -> tuple[TurnScore, GraphStep]:
    """Record an answer, resume the graph, and persist the resulting score."""
    if interview.status is not InterviewStatus.in_progress:
        raise InterviewError("This interview is no longer active.")

    pending = await db.scalar(
        select(Turn)
        .where(Turn.interview_id == interview.id, Turn.answer_text.is_(None))
        .order_by(Turn.index.desc())
        .limit(1)
    )
    if pending is None:
        raise InterviewError("There is no question waiting for an answer.")

    step = await _advance(
        thread_config(interview.thread_id),
        Command(
            resume={
                "answer": answer,
                "word_timings": word_timings,
                "audio_duration_s": audio_duration_s,
            }
        ),
    )

    graph_turns = (step.state or {}).get("turns") or []
    scored = graph_turns[-1] if graph_turns else {}

    pending.answer_text = answer
    pending.answered_at = datetime.now(UTC)
    pending.audio_duration_s = audio_duration_s
    pending.transcription_confidence = transcription_confidence
    pending.word_timings = word_timings

    score = TurnScore(
        turn_id=pending.id,
        model_score=float(scored.get("model_score", 0.0)),
        llm_score=scored.get("llm_score"),
        relevance=scored.get("relevance"),
        blended_score=float(scored.get("overall", 0.0)),
        structure=scored.get("structure"),
        specificity=scored.get("specificity"),
        clarity=scored.get("clarity"),
        depth=scored.get("depth"),
        features=scored.get("features", {}),
        rubric_notes=scored.get("notes"),
        scorer_version=SCORER_VERSION,
    )
    db.add(score)

    if step.finished:
        await _finalise(db, interview, step, graph_turns)
    elif step.question:
        db.add(
            Turn(
                interview_id=interview.id,
                index=step.turn_index,
                kind=step.kind,
                competency=step.competency,
                question=step.question,
            )
        )

    await db.commit()
    await db.refresh(score)
    return score, step


async def _finalise(
    db: AsyncSession,
    interview: Interview,
    step: GraphStep,
    graph_turns: list[dict[str, Any]],
) -> None:
    """Write the report and close the interview."""
    report_data = step.report or {}
    delivery = build_report_delivery([t["features"] for t in graph_turns if t.get("features")])

    db.add(
        InterviewReport(
            interview_id=interview.id,
            overall_score=float(report_data.get("overall_score", 0.0)),
            competency_scores=report_data.get("competency_scores", {}),
            delivery_metrics=delivery,
            summary=report_data.get("summary", ""),
            strengths=report_data.get("strengths", []),
            improvements=report_data.get("improvements", []),
            recommended_focus=report_data.get("recommended_focus"),
        )
    )
    interview.status = InterviewStatus.completed
    interview.completed_at = datetime.now(UTC)
    log.info("interview.completed", interview_id=str(interview.id))


async def get_interview(db: AsyncSession, user: User, interview_id: uuid.UUID) -> Interview:
    interview = await db.scalar(
        select(Interview).where(Interview.id == interview_id, Interview.user_id == user.id)
    )
    if interview is None:
        raise InterviewError("Interview not found.")
    return interview


async def list_interviews(
    db: AsyncSession, user: User, *, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    """History rows with answered-question counts and overall score, in one query."""
    answered = (
        select(Turn.interview_id, func.count(Turn.id).label("answered"))
        .where(Turn.answer_text.isnot(None))
        .group_by(Turn.interview_id)
        .subquery()
    )

    rows = await db.execute(
        select(
            Interview,
            func.coalesce(answered.c.answered, 0).label("answered_questions"),
            InterviewReport.overall_score,
        )
        .outerjoin(answered, answered.c.interview_id == Interview.id)
        .outerjoin(InterviewReport, InterviewReport.interview_id == Interview.id)
        .where(Interview.user_id == user.id)
        .order_by(Interview.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    return [
        {
            "id": interview.id,
            "role": interview.role,
            "seniority": interview.seniority,
            "persona": interview.persona,
            "status": interview.status,
            "created_at": interview.created_at,
            "completed_at": interview.completed_at,
            "answered_questions": int(answered_count),
            "overall_score": float(score) if score is not None else None,
        }
        for interview, answered_count, score in rows.all()
    ]


async def abandon_interview(db: AsyncSession, interview: Interview) -> Interview:
    if interview.status is InterviewStatus.in_progress:
        interview.status = InterviewStatus.abandoned
        interview.completed_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(interview)
    return interview


async def current_question(db: AsyncSession, interview: Interview) -> Turn | None:
    """The question awaiting an answer, if any. Used to resume after a reload."""
    return await db.scalar(
        select(Turn)
        .where(Turn.interview_id == interview.id, Turn.answer_text.is_(None))
        .order_by(Turn.index.desc())
        .limit(1)
    )
