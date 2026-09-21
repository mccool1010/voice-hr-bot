"""Groq provider — free hosted tier, very high throughput.

Groq serves open models on custom silicon; token latency is low enough that a
voice interview feels responsive without a local GPU. This is the default for
deployed environments where no Anthropic key is configured.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from groq import AsyncStream
    from groq.types.chat import ChatCompletionChunk

from app.llm.base import BaseLLMProvider, ChatMessage, LLMError, LLMUnavailableError


class GroqProvider(BaseLLMProvider):
    name = "groq"

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b") -> None:
        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailableError("groq package is not installed") from exc

        if not api_key:
            raise LLMUnavailableError("GROQ_API_KEY is not set")

        self.model = model
        self._client = AsyncGroq(api_key=api_key, max_retries=3)

    def _payload(self, system: str, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, *(m.model_dump() for m in messages)]

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        import groq

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=self._payload(system, messages),  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except groq.RateLimitError as exc:
            raise LLMError("Groq rate limit reached — retry shortly.") from exc
        except groq.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach Groq.") from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq API error ({exc.status_code})") from exc

        return (response.choices[0].message.content or "").strip()

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        import groq

        try:
            # stream=True always yields a stream; the SDK's return type is a union.
            stream = cast(
                "AsyncStream[ChatCompletionChunk]",
                await self._client.chat.completions.create(
                    model=self.model,
                    messages=self._payload(system, messages),  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                ),
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except groq.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach Groq.") from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq API error ({exc.status_code})") from exc

    async def health(self) -> bool:
        try:
            await self._client.models.list()
        except Exception:
            return False
        return True
