"""The LangGraph interview loop, driven end to end with no network.

These tests exercise the real compiled graph — interrupts, checkpointing,
conditional routing and the scoring pipeline — against `EchoProvider`.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.builder import build_graph, thread_config
from app.graph.contracts import (
    AnswerEvaluation,
    InterviewPlan,
    NextQuestion,
    PlannedTopic,
    ReportDraft,
)
from app.graph.state import MAX_PROBE_DEPTH, initial_state
from app.llm import EchoProvider, set_provider
from app.models.enums import Competency, Persona, Seniority
from app.scoring import service


@pytest.fixture(autouse=True)
def offline_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.settings, "embeddings_enabled", False)


def _plan(n: int) -> InterviewPlan:
    competencies = list(Competency)
    return InterviewPlan(
        opening_remark="Welcome — thanks for joining.",
        topics=[
            PlannedTopic(
                title=f"Topic {i}",
                competency=competencies[i % len(competencies)],
                rationale="Relevant to the role.",
                opening_question=f"Opening question {i}?",
            )
            for i in range(n)
        ],
    )


def _evaluation(*, probe: bool, overall: float = 70.0) -> AnswerEvaluation:
    return AnswerEvaluation(
        structure=overall,
        specificity=overall,
        clarity=overall,
        depth=overall,
        overall=overall,
        notes="Fine.",
        should_probe=probe,
        probe_focus="the metrics they cited" if probe else None,
    )


def _report() -> ReportDraft:
    return ReportDraft(
        summary="You did well.",
        strengths=["Concrete examples"],
        improvements=["Quantify impact"],
        recommended_focus="Practise the STAR structure.",
    )


def _pending(result: dict[str, Any]) -> dict[str, Any] | None:
    raw = result.get("__interrupt__")
    if not raw:
        return None
    return getattr(raw[0], "value", raw[0])


async def _start(provider: EchoProvider, topics: int = 2) -> tuple[Any, dict[str, Any], dict]:
    set_provider(provider)
    graph = build_graph(InMemorySaver())
    config = thread_config("test-thread")
    result = await graph.ainvoke(
        initial_state(
            interview_id="i-1",
            role="Backend Engineer",
            seniority=Seniority.mid,
            persona=Persona.neutral,
            target_questions=topics,
        ),
        config=config,
    )
    return graph, config, result


async def test_graph_asks_planned_opening_question_first() -> None:
    _, _, result = await _start(EchoProvider(structured_responses=[_plan(2)]))
    pending = _pending(result)

    assert pending is not None
    assert pending["question"] == "Opening question 0?"
    assert result["opening_remark"] == "Welcome — thanks for joining."
    set_provider(None)


async def test_full_interview_runs_to_a_report() -> None:
    provider = EchoProvider(
        structured_responses=[
            _plan(2),
            _evaluation(probe=False, overall=80.0),
            _evaluation(probe=False, overall=60.0),
            _report(),
        ]
    )
    graph, config, result = await _start(provider, topics=2)

    result = await graph.ainvoke(
        Command(resume={"answer": "I built a payments retry layer in Redis."}), config=config
    )
    assert _pending(result)["question"] == "Opening question 1?"

    result = await graph.ainvoke(
        Command(resume={"answer": "We migrated auth to a separate service."}), config=config
    )

    assert _pending(result) is None
    assert result["finished"] is True
    report = result["report"]
    assert report["summary"] == "You did well."
    assert len(result["turns"]) == 2
    assert 0 < report["overall_score"] <= 100
    assert set(report["competency_scores"]) == {"communication", "technical_depth"}
    set_provider(None)


async def test_probe_generates_follow_up_on_same_topic() -> None:
    provider = EchoProvider(
        structured_responses=[
            _plan(2),
            _evaluation(probe=True),
            NextQuestion(
                question="What drove the 94% figure?", competency=Competency.communication
            ),
        ]
    )
    graph, config, _ = await _start(provider)

    result = await graph.ainvoke(
        Command(resume={"answer": "Timeouts dropped by 94 percent."}), config=config
    )
    pending = _pending(result)

    assert pending["question"] == "What drove the 94% figure?"
    assert pending["kind"] == "follow_up"
    set_provider(None)


async def test_probe_depth_is_capped() -> None:
    """The interviewer must move on even if the evaluator always wants more."""
    evaluations = [_evaluation(probe=True) for _ in range(MAX_PROBE_DEPTH + 2)]
    provider = EchoProvider(structured_responses=[_plan(2), *evaluations])
    graph, config, _ = await _start(provider)

    questions = []
    for _ in range(MAX_PROBE_DEPTH + 1):
        result = await graph.ainvoke(Command(resume={"answer": "An answer."}), config=config)
        questions.append(_pending(result)["question"])

    # After MAX_PROBE_DEPTH follow-ups the graph advances to topic 1.
    assert questions[-1] == "Opening question 1?"
    set_provider(None)


async def test_state_is_checkpointed_between_invocations() -> None:
    graph, config, _ = await _start(EchoProvider(structured_responses=[_plan(3)]))
    snapshot = await graph.aget_state(config)

    assert snapshot.next == ("await_answer",)
    assert snapshot.values["current_question"] == "Opening question 0?"
    set_provider(None)


async def test_evaluator_failure_degrades_to_local_model() -> None:
    from app.llm.base import LLMError

    class FlakyEvaluator(EchoProvider):
        async def structured(self, *, schema, **kwargs):  # type: ignore[no-untyped-def,override]
            if schema is AnswerEvaluation:
                raise LLMError("evaluator down")
            return await super().structured(schema=schema, **kwargs)

    provider = FlakyEvaluator(structured_responses=[_plan(1)])
    graph, config, _ = await _start(provider, topics=1)
    result = await graph.ainvoke(
        Command(resume={"answer": "I designed the caching layer and cut p99 by 40ms."}),
        config=config,
    )

    turn = result["turns"][-1]
    assert turn["overall"] == pytest.approx(turn["model_score"])
    assert result["finished"] is True
    set_provider(None)


async def test_graph_survives_enums_restored_as_strings() -> None:
    """Regression: the Postgres checkpointer returns enums as plain strings.

    The in-memory saver keeps enum objects, which hid a crash on resume. Feeding
    plain strings reproduces what a real restored checkpoint looks like.
    """
    set_provider(EchoProvider(structured_responses=[_plan(1), _evaluation(probe=False)]))
    graph = build_graph(InMemorySaver())
    config = thread_config("string-enums")
    state = initial_state(
        interview_id="i-2",
        role="SRE",
        seniority=Seniority.senior,
        persona=Persona.skeptical,
        target_questions=1,
    )
    state["seniority"] = "senior"  # type: ignore[typeddict-item]
    state["persona"] = "skeptical"  # type: ignore[typeddict-item]

    await graph.ainvoke(state, config=config)
    result = await graph.ainvoke(Command(resume={"answer": "I own on-call."}), config=config)

    assert result["finished"] is True
    set_provider(None)
