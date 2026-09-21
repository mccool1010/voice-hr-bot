"""Groq per-model rate-limit failover and reasoning-effort routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from app.llm.base import ChatMessage, LLMError
from app.llm.groq_provider import GroqProvider

PRIMARY = "openai/gpt-oss-120b"
FALLBACK = "qwen/qwen3.8-27b"


def _error(cls: type[groq.APIStatusError], status: int) -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("error", response=httpx.Response(status, request=request), body=None)


class FakeCompletions:
    def __init__(self, failures: dict[str, groq.APIStatusError]) -> None:
        self.failures = failures
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs["model"] in self.failures:
            raise self.failures[kwargs["model"]]
        message = SimpleNamespace(content=f"answer from {kwargs['model']}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _provider(failures: dict[str, groq.APIStatusError]) -> tuple[GroqProvider, FakeCompletions]:
    provider = GroqProvider(api_key="gsk_test", model=PRIMARY, fallback_models=[FALLBACK])
    completions = FakeCompletions(failures)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider._fast = fake  # type: ignore[assignment]
    provider._patient = fake  # type: ignore[assignment]
    return provider, completions


async def _ask(provider: GroqProvider) -> str:
    return await provider.complete(system="s", messages=[ChatMessage(role="user", content="hi")])


async def test_primary_is_used_when_available() -> None:
    provider, calls = _provider({})
    assert await _ask(provider) == f"answer from {PRIMARY}"
    assert [c["model"] for c in calls.calls] == [PRIMARY]


async def test_rate_limited_primary_fails_over_immediately() -> None:
    provider, calls = _provider({PRIMARY: _error(groq.RateLimitError, 429)})
    assert await _ask(provider) == f"answer from {FALLBACK}"
    assert [c["model"] for c in calls.calls] == [PRIMARY, FALLBACK]


async def test_every_model_limited_raises() -> None:
    provider, _ = _provider(
        {PRIMARY: _error(groq.RateLimitError, 429), FALLBACK: _error(groq.RateLimitError, 429)}
    )
    with pytest.raises(LLMError, match="rate limit"):
        await _ask(provider)


async def test_other_errors_do_not_fail_over() -> None:
    """A 400 is a bug in the request; another model would reject it too."""
    provider, calls = _provider({PRIMARY: _error(groq.BadRequestError, 400)})
    with pytest.raises(LLMError, match="400"):
        await _ask(provider)
    assert [c["model"] for c in calls.calls] == [PRIMARY]


async def test_reasoning_effort_is_only_sent_to_gpt_oss() -> None:
    provider, calls = _provider({PRIMARY: _error(groq.RateLimitError, 429)})
    await _ask(provider)
    primary_call, fallback_call = calls.calls
    assert primary_call["reasoning_effort"] == "low"
    # It would switch Qwen's reasoning *on* and cost more tokens.
    assert "reasoning_effort" not in fallback_call
