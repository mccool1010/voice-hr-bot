"""Claude provider — highest reasoning quality, used for the hosted demo.

Two things here are worth the vendor-specific code: adaptive thinking (better
follow-up questions and more consistent rubric grading) and prompt caching on the
system block, which is reused unchanged on every turn of an interview.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from pydantic import BaseModel

from app.llm.base import BaseLLMProvider, ChatMessage, LLMError, LLMUnavailableError

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        MessageParam,
        OutputConfigParam,
        TextBlockParam,
        ThinkingConfigParam,
    )

# Adaptive thinking: the model decides per request how much to reason. It pays
# off in follow-up generation and keeps rubric grading consistent.
_THINKING: ThinkingConfigParam = {"type": "adaptive"}

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-opus-5", effort: str = "high") -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailableError("anthropic package is not installed") from exc

        if not api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")

        self.model = model
        self.effort = effort
        self._client: AsyncAnthropic = AsyncAnthropic(api_key=api_key, max_retries=3)

    def _system_blocks(self, system: str) -> list[TextBlockParam]:
        # The system prompt is byte-identical across an interview's turns, so it
        # is the natural cache breakpoint. Volatile content stays in `messages`.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    @staticmethod
    def _messages(messages: Sequence[ChatMessage]) -> list[MessageParam]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @property
    def _output_config(self) -> OutputConfigParam:
        return cast("OutputConfigParam", {"effort": self.effort})

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        import anthropic

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self._system_blocks(system),
                messages=self._messages(messages),
                thinking=_THINKING,
                output_config=self._output_config,
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit reached — retry shortly.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach the Claude API.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error ({exc.status_code}): {exc.message}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this prompt.")

        return "".join(b.text for b in response.content if b.type == "text").strip()

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        import anthropic

        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=self._system_blocks(system),
                messages=self._messages(messages),
                thinking=_THINKING,
                output_config=self._output_config,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach the Claude API.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error ({exc.status_code}): {exc.message}") from exc

    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        schema: type[T],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> T:
        """Native structured output — the SDK validates against the schema."""
        import anthropic

        try:
            response = await self._client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=self._system_blocks(system),
                messages=self._messages(messages),
                output_format=schema,
            )
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach the Claude API.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error ({exc.status_code}): {exc.message}") from exc

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"Claude returned no parsable {schema.__name__}")
        return parsed

    async def health(self) -> bool:
        try:
            await self._client.models.retrieve(self.model)
        except Exception:
            return False
        return True
