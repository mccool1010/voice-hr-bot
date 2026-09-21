"""LangGraph nodes for the interview loop.

Each node is a pure-ish async function: state in, partial state out. Nothing
here touches the database — persistence is the caller's job, which keeps the
graph testable against `EchoProvider` with no infrastructure at all.

Flow:
    plan → ask → (interrupt for the answer) → evaluate → route
    route → ask        when probing deeper or moving to the next topic
    route → report     when the agenda is exhausted
"""

from __future__ import annotations

import structlog
from langgraph.types import interrupt

from app.graph.contracts import (
    AnswerEvaluation,
    InterviewPlan,
    NextQuestion,
    PlannedTopic,
    ReportDraft,
)
from app.graph.state import MAX_PROBE_DEPTH, InterviewState, TurnRecord
from app.llm import ChatMessage, LLMError, get_provider
from app.llm.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    interviewer_system_prompt,
    planner_system_prompt,
    transcript_digest,
)
from app.models.enums import Competency, Persona, Seniority, TurnKind
from app.scoring import service as scoring

log = structlog.get_logger(__name__)


def _history(state: InterviewState, limit: int = 10) -> list[ChatMessage]:
    """Recent transcript as alternating chat turns."""
    messages: list[ChatMessage] = []
    for turn in (state.get("turns") or [])[-limit:]:
        messages.append(ChatMessage(role="assistant", content=turn.get("question", "")))
        if answer := turn.get("answer"):
            messages.append(ChatMessage(role="user", content=answer))
    return messages


def _seniority(state: InterviewState) -> Seniority:
    """Enum-typed seniority.

    The Postgres checkpointer serialises state, so enums come back as plain
    strings after a resume. Coerce at the boundary instead of trusting the type.
    """
    return Seniority(state["seniority"])


def _persona(state: InterviewState) -> Persona:
    return Persona(state["persona"])


def _system_for(state: InterviewState) -> str:
    return interviewer_system_prompt(
        role=state["role"],
        seniority=_seniority(state),
        persona=_persona(state),
        resume_context=state.get("resume_context"),
    )


def _topics(state: InterviewState) -> list[dict[str, object]]:
    plan = state.get("plan") or {}
    return list(plan.get("topics") or [])


# ─── Nodes ────────────────────────────────────────────────────────────────────


async def plan_interview(state: InterviewState) -> dict[str, object]:
    """Produce the agenda once, up front.

    Planning ahead rather than improvising each question is what keeps the
    interview balanced across competencies — an LLM asked only for "the next
    question" drifts toward whatever the candidate last mentioned.
    """
    provider = get_provider()
    target = state.get("target_questions", 6)

    try:
        plan = await provider.structured(
            system=planner_system_prompt(
                role=state["role"],
                seniority=_seniority(state),
                topic_count=target,
                resume_context=state.get("resume_context"),
            ),
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        f"Design the agenda for a {_seniority(state).value}-level "
                        f"{state['role']} interview with exactly {target} topics."
                    ),
                )
            ],
            schema=InterviewPlan,
            max_tokens=2048,
        )
    except LLMError as exc:
        log.error("graph.plan_failed", error=str(exc))
        return {"error": f"Could not plan the interview: {exc}", "finished": True}

    log.info("graph.planned", topics=len(plan.topics), interview=state.get("interview_id"))
    return {
        "plan": plan.model_dump(mode="json"),
        "opening_remark": plan.opening_remark,
        "topic_index": 0,
        "probe_depth": 0,
    }


async def ask_question(state: InterviewState) -> dict[str, object]:
    """Emit the next question.

    Two paths: the scripted opening question for a fresh topic (cheap, already
    planned) or a generated follow-up that reacts to what was just said.
    """
    topics = _topics(state)
    topic_index = state.get("topic_index", 0)
    probe_depth = state.get("probe_depth", 0)
    turn_index = state.get("turn_index", 0)

    if topic_index >= len(topics):
        return {"finished": True}

    topic = PlannedTopic.model_validate(topics[topic_index])

    # Fresh topic — use the planned question verbatim. No LLM call needed.
    if probe_depth == 0:
        return {
            "current_question": topic.opening_question,
            "current_competency": topic.competency,
            "current_kind": TurnKind.opening if turn_index == 0 else TurnKind.main,
        }

    # Follow-up — generate something that actually responds to the last answer.
    turns = state.get("turns") or []
    last = turns[-1] if turns else {}
    focus = last.get("probe_focus") or "the part of their answer that was least concrete"

    provider = get_provider()
    try:
        nxt = await provider.structured(
            system=_system_for(state),
            messages=[
                *_history(state, limit=6),
                ChatMessage(
                    role="user",
                    content=(
                        f"[Director's note — not from the candidate] Ask one follow-up "
                        f"about: {focus}. Stay on the topic '{topic.title}'. Do not "
                        f"restate their answer back to them."
                    ),
                ),
            ],
            schema=NextQuestion,
            max_tokens=512,
        )
        question, competency = nxt.question, nxt.competency
    except LLMError as exc:
        log.warning("graph.followup_failed", error=str(exc))
        question = f"Could you say more about {focus}?"
        competency = topic.competency

    return {
        "current_question": question,
        "current_competency": competency,
        "current_kind": TurnKind.follow_up,
    }


async def await_answer(state: InterviewState) -> dict[str, object]:
    """Human-in-the-loop pause.

    `interrupt` suspends the graph and checkpoints it. The transport layer
    resumes with `Command(resume=...)` once the candidate has spoken, which is
    what makes an interview survive a dropped WebSocket or a page reload.
    """
    payload = interrupt(
        {
            "type": "question",
            "question": state.get("current_question", ""),
            "competency": str(state.get("current_competency", Competency.communication)),
            "kind": str(state.get("current_kind", TurnKind.main)),
            "turn_index": state.get("turn_index", 0),
        }
    )

    if isinstance(payload, str):
        payload = {"answer": payload}
    payload = payload or {}

    return {
        "_answer": payload.get("answer", ""),
        "_word_timings": payload.get("word_timings"),
        "_audio_duration_s": payload.get("audio_duration_s"),
    }


async def evaluate_answer(state: InterviewState) -> dict[str, object]:
    """Grade the answer with the LLM rubric, then blend in the local signals."""
    question = state.get("current_question", "")
    answer = str(state.get("_answer") or "")
    word_timings = state.get("_word_timings")
    audio_duration = state.get("_audio_duration_s")

    llm_scores: dict[str, float] | None = None
    notes = ""
    should_probe = False
    probe_focus: str | None = None

    if answer.strip():
        provider = get_provider()
        try:
            evaluation = await provider.structured(
                system=EVALUATOR_SYSTEM_PROMPT,
                messages=[
                    ChatMessage(
                        role="user",
                        content=(
                            f"Role: {state['role']} ({_seniority(state).value})\n"
                            f"Competency probed: {state.get('current_competency')}\n\n"
                            f"Question: {question}\n\n"
                            f"Answer: {answer}"
                        ),
                    )
                ],
                schema=AnswerEvaluation,
                max_tokens=1024,
            )
            llm_scores = {
                "structure": evaluation.structure,
                "specificity": evaluation.specificity,
                "clarity": evaluation.clarity,
                "depth": evaluation.depth,
                "overall": evaluation.overall,
            }
            notes = evaluation.notes
            should_probe = evaluation.should_probe
            probe_focus = evaluation.probe_focus
        except LLMError as exc:
            # Degrade to the local model rather than losing the turn entirely.
            log.warning("graph.evaluation_failed", error=str(exc))

    result = await scoring.score_answer(
        question=question,
        answer=answer,
        llm_scores=llm_scores,
        word_timings=word_timings,
        audio_duration_s=audio_duration,
    )

    turn: TurnRecord = {
        "index": state.get("turn_index", 0),
        "question": question,
        "answer": answer,
        "competency": str(state.get("current_competency", Competency.communication)),
        "kind": str(state.get("current_kind", TurnKind.main)),
        "overall": result.blended_score,
        "structure": result.structure or 0.0,
        "specificity": result.specificity or 0.0,
        "clarity": result.clarity or 0.0,
        "depth": result.depth or 0.0,
        "notes": notes,
        "model_score": result.model_score,
        "llm_score": result.llm_score,
        "relevance": result.relevance,
        "features": result.features,
    }
    # Consumed by `ask_question` when it generates a follow-up.
    turn["probe_focus"] = probe_focus

    return {
        "turns": [turn],
        "turn_index": state.get("turn_index", 0) + 1,
        "_should_probe": should_probe,
        "_answer": None,
        "_word_timings": None,
        "_audio_duration_s": None,
    }


async def advance(state: InterviewState) -> dict[str, object]:
    """Move the cursor: probe the same topic again, or step to the next one."""
    topics = _topics(state)
    topic_index = state.get("topic_index", 0)
    probe_depth = state.get("probe_depth", 0)

    wants_probe = bool(state.get("_should_probe"))
    can_probe = probe_depth < MAX_PROBE_DEPTH

    if wants_probe and can_probe:
        return {"probe_depth": probe_depth + 1, "_should_probe": False}

    next_topic = topic_index + 1
    return {
        "topic_index": next_topic,
        "probe_depth": 0,
        "_should_probe": False,
        "finished": next_topic >= len(topics),
    }


async def compose_report(state: InterviewState) -> dict[str, object]:
    """Write the closing feedback and roll up competency scores."""
    turns = state.get("turns") or []
    answered = [t for t in turns if (t.get("answer") or "").strip()]

    if not answered:
        return {
            "report": {
                "summary": "The interview ended before any questions were answered.",
                "strengths": [],
                "improvements": ["Complete an interview to receive feedback."],
                "recommended_focus": None,
                "overall_score": 0.0,
                "competency_scores": {},
            },
            "finished": True,
        }

    overall = sum(float(t.get("overall", 0.0)) for t in answered) / len(answered)

    competency_scores: dict[str, float] = {}
    for competency in {str(t.get("competency")) for t in answered}:
        matching = [
            float(t.get("overall", 0.0)) for t in answered if str(t.get("competency")) == competency
        ]
        competency_scores[competency] = round(sum(matching) / len(matching), 2)

    provider = get_provider()
    digest = transcript_digest([(t.get("question", ""), t.get("answer", "")) for t in answered])

    try:
        draft = await provider.structured(
            system=REPORTER_SYSTEM_PROMPT,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        f"Role: {state['role']} ({_seniority(state).value})\n\n"
                        f"Transcript:\n{digest}\n\n"
                        "Write the candidate's closing feedback."
                    ),
                )
            ],
            schema=ReportDraft,
            max_tokens=2048,
        )
        report = draft.model_dump(mode="json")
    except LLMError as exc:
        log.warning("graph.report_failed", error=str(exc))
        report = {
            "summary": (
                "Your interview is complete. Detailed written feedback could not be "
                "generated, but your per-answer scores are shown below."
            ),
            "strengths": [],
            "improvements": [],
            "recommended_focus": None,
        }

    report |= {
        "overall_score": round(overall, 2),
        "competency_scores": competency_scores,
    }
    log.info("graph.report_composed", overall=round(overall, 2), turns=len(answered))
    return {"report": report, "finished": True}


# ─── Conditional edges ────────────────────────────────────────────────────────


def route_after_advance(state: InterviewState) -> str:
    """Continue the agenda or wrap up."""
    if state.get("error"):
        return "report"
    return "report" if state.get("finished") else "ask"


def route_after_plan(state: InterviewState) -> str:
    """Skip straight to the report if planning failed."""
    return "report" if state.get("error") or state.get("finished") else "ask"
