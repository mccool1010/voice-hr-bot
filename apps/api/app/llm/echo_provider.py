"""Deterministic in-process provider used by the test suite and CI.

No network, no model weights, no API key. It lets the whole interview graph,
the scoring pipeline and every route be tested end to end in CI without
provisioning inference. Tests can script exact replies; anything unscripted
falls back to a stable synthetic answer derived from the schema.
"""

from __future__ import annotations

import re
import typing
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pydantic import BaseModel

from app.llm.base import BaseLLMProvider, ChatMessage

T = TypeVar("T", bound=BaseModel)


def _synthesise(annotation: object) -> object:
    """Build a type-appropriate placeholder for a required field."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    # Optional[X] / X | None — a null satisfies the schema.
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        if type(None) in args:
            return None
        return _synthesise(args[0])
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 50.0
    if annotation is str:
        return "synthetic"
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _build(annotation)
        # StrEnum and friends — take the first member.
        members = getattr(annotation, "__members__", None)
        if members:
            return next(iter(members.values()))
    return "synthetic"


def _min_length(metadata: list[object]) -> int:
    for item in metadata:
        if (value := getattr(item, "min_length", None)) is not None:
            return int(value)
    return 0


def _build(schema: type[T]) -> T:
    values: dict[str, object] = {}
    for field_name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        value = _synthesise(field.annotation)
        # An empty list would violate `min_length`; fill it with valid items.
        if isinstance(value, list) and (count := _min_length(field.metadata)):
            item_args = typing.get_args(field.annotation)
            value = [_synthesise(item_args[0] if item_args else str) for _ in range(count)]
        values[field_name] = value
    return schema.model_validate(values)


_CANNED_QUESTIONS = (
    (
        "Background",
        "communication",
        "To start, walk me through your background and what drew you to this role.",
    ),
    (
        "Recent project",
        "technical_depth",
        "Tell me about a recent project you're proud of. What was your part in it?",
    ),
    (
        "Hard problem",
        "problem_solving",
        "Describe the hardest technical problem you've solved. How did you approach it?",
    ),
    (
        "Ownership",
        "ownership",
        "Tell me about a time something you owned went wrong. What did you do?",
    ),
    (
        "Collaboration",
        "collaboration",
        "Describe a disagreement with a teammate and how it was resolved.",
    ),
    ("Growth", "culture_fit", "What are you trying to get better at right now, and how?"),
)


_TOPIC_COUNT = re.compile(r"exactly (\d+) topics")


def _canned(schema: type[BaseModel], system: str) -> BaseModel | None:
    """Readable defaults for the interview plan, so echo mode is demo-able.

    Imported lazily: the graph package imports this module, so importing the
    contracts at module level would be circular.
    """
    from app.graph.contracts import InterviewPlan, PlannedTopic, ReportDraft

    if schema is ReportDraft:
        # Say plainly that no model wrote this, rather than faking feedback.
        return ReportDraft(
            summary=(
                "This interview ran in offline echo mode, so no language model reviewed "
                "your answers. The scores come from the local PyTorch delivery model only. "
                "Configure Ollama, Groq or Claude for written feedback."
            ),
            strengths=["Written feedback needs a language model; see the per-answer scores."],
            improvements=["Written feedback needs a language model; see the per-answer scores."],
            recommended_focus="Connect an LLM provider to get personalised coaching.",
        )

    if schema is InterviewPlan:
        # Honour the requested length, as a real model would.
        match = _TOPIC_COUNT.search(system)
        count = max(1, int(match.group(1))) if match else len(_CANNED_QUESTIONS)
        return InterviewPlan(
            opening_remark="Thanks for joining. Let's get started.",
            topics=[
                PlannedTopic(
                    title=title,
                    competency=competency,
                    rationale="Standard practice question.",
                    opening_question=question,
                )
                for title, competency, question in _CANNED_QUESTIONS[:count]
            ],
        )
    return None


class EchoProvider(BaseLLMProvider):
    name = "echo"

    def __init__(
        self,
        model: str = "echo-1",
        scripted: Sequence[str] | None = None,
        structured_responses: Sequence[BaseModel] | None = None,
    ) -> None:
        self.model = model
        self._scripted = list(scripted or [])
        self._structured = list(structured_responses or [])
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append({"system": system, "messages": [m.model_dump() for m in messages]})
        if self._scripted:
            return self._scripted.pop(0)
        last = messages[-1].content if messages else ""
        return f"Thanks for that. Could you tell me more about {last[:60].strip()}?"

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        text = await self.complete(
            system=system, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        for word in text.split(" "):
            yield word + " "

    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        schema: type[T],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> T:
        self.calls.append({"system": system, "schema": schema.__name__})
        # Take the first scripted response of the right type; leave the rest
        # queued, so scripts for different schemas can interleave freely.
        for i, candidate in enumerate(self._structured):
            if isinstance(candidate, schema):
                return self._structured.pop(i)  # type: ignore[return-value]
        if (canned := _canned(schema, system)) is not None:
            return canned  # type: ignore[return-value]
        return _build(schema)

    async def health(self) -> bool:
        return True
