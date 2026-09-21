"""Provider-agnostic LLM interface.

Every interview node talks to this protocol, never to a vendor SDK. Swapping
Ollama for Claude is an environment variable, not a code change.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

Role = Literal["user", "assistant"]

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class ChatMessage(BaseModel):
    role: Role
    content: str


class LLMError(RuntimeError):
    """Raised when a provider fails in a way the caller should surface."""


class LLMUnavailableError(LLMError):
    """Provider is unreachable or unconfigured — callers may fall back."""


def extract_json(raw: str) -> str:
    """Pull a JSON object out of a model response.

    Providers without native schema enforcement wrap JSON in prose or fences.
    This strips both without being clever enough to corrupt valid JSON.
    """
    fenced = _FENCE_RE.match(raw)
    if fenced:
        return fenced.group(1)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


class BaseLLMProvider(ABC):
    """Shared behaviour. Subclasses implement the three transport methods."""

    name: str
    model: str

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Return a single completion as plain text."""

    @abstractmethod
    def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""

    @abstractmethod
    async def health(self) -> bool:
        """True when the provider is reachable and configured."""

    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        schema: type[T],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> T:
        """Return a validated Pydantic model.

        The default implementation prompts for JSON and validates the result,
        retrying once with the validation error fed back. Providers with native
        schema enforcement override this.
        """
        instruction = (
            f"{system}\n\n"
            "Respond with a single JSON object and nothing else — no prose, no "
            "markdown fences. It must match this JSON Schema exactly:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        attempt_messages = list(messages)

        for attempt in range(2):
            raw = await self.complete(
                system=instruction,
                messages=attempt_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as exc:
                if attempt == 1:
                    raise LLMError(
                        f"{self.name} did not return valid {schema.__name__}: {exc}"
                    ) from exc
                attempt_messages = [
                    *messages,
                    ChatMessage(role="assistant", content=raw[:2000]),
                    ChatMessage(
                        role="user",
                        content=(
                            f"That was not valid. The validator reported:\n{exc}\n\n"
                            "Return only the corrected JSON object."
                        ),
                    ),
                ]
        raise LLMError("unreachable")  # pragma: no cover
