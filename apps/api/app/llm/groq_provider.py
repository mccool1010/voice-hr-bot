"""Groq provider — free hosted tier, very high throughput.

Groq serves open models on custom silicon; token latency is low enough that a
voice interview feels responsive without a local GPU. This is the default for
deployed environments where no Anthropic key is configured.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import structlog

if TYPE_CHECKING:
    from groq import AsyncStream
    from groq.types.chat import ChatCompletionChunk

from app.llm.base import BaseLLMProvider, ChatMessage, LLMError, LLMUnavailableError

log = structlog.get_logger(__name__)


class GroqProvider(BaseLLMProvider):
    """Groq chat with per-model rate-limit failover.

    Groq's free tier limits tokens per minute *per model*. Rather than sleep on
    a 429's retry-after (up to ~10 s, which stalls a live interview), each call
    walks the model chain: primary first, then each fallback. Only the last
    model in the chain waits and retries.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        *,
        fallback_models: Sequence[str] = (),
        reasoning_effort: str | None = "low",
    ) -> None:
        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailableError("groq package is not installed") from exc

        if not api_key:
            raise LLMUnavailableError("GROQ_API_KEY is not set")

        self.model = model
        self.models = [model, *(m for m in fallback_models if m != model)]
        self.reasoning_effort = reasoning_effort
        # No automatic retries while another model can take the request; the
        # final model in the chain retries and honours retry-after.
        self._fast = AsyncGroq(api_key=api_key, max_retries=0)
        self._patient = AsyncGroq(api_key=api_key, max_retries=3)

    def _payload(self, system: str, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, *(m.model_dump() for m in messages)]

    def _extra(self, model: str) -> dict[str, str]:
        # reasoning_effort lowers gpt-oss token use, but *turns on* reasoning
        # for Qwen (measured: 2 → 20 completion tokens), so only send it to gpt-oss.
        if self.reasoning_effort and model.startswith("openai/gpt-oss"):
            return {"reasoning_effort": self.reasoning_effort}
        return {}

    async def _create(self, **kwargs: Any) -> Any:
        import groq

        for i, model in enumerate(self.models):
            last = i == len(self.models) - 1
            client = self._patient if last else self._fast
            try:
                # kwargs are forwarded from typed callers above; mypy cannot
                # match the SDK's stream/non-stream overloads through **kwargs.
                return await client.chat.completions.create(  # type: ignore[call-overload]
                    model=model, **kwargs, **self._extra(model)
                )
            except groq.RateLimitError as exc:
                if last:
                    raise LLMError("Groq rate limit reached — retry shortly.") from exc
                log.info("llm.groq.rate_limited_failover", model=model, next=self.models[i + 1])
            except groq.APIConnectionError as exc:
                raise LLMUnavailableError("Could not reach Groq.") from exc
            except groq.APIStatusError as exc:
                raise LLMError(f"Groq API error ({exc.status_code})") from exc
        raise LLMError("No Groq model available.")  # pragma: no cover

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        response = await self._create(
            messages=self._payload(system, messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        # stream=True always yields a stream; the SDK's return type is a union.
        stream = cast(
            "AsyncStream[ChatCompletionChunk]",
            await self._create(
                messages=self._payload(system, messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ),
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def health(self) -> bool:
        try:
            await self._fast.models.list()
        except Exception:
            return False
        return True
