"""Ollama provider — local inference, no rate limits, no per-token cost.

This is the development default. Ollama exposes native JSON-schema constrained
decoding through the `format` field, so `structured()` is enforced by the
runtime rather than by prompt discipline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.llm.base import BaseLLMProvider, ChatMessage, LLMError, LLMUnavailableError

T = TypeVar("T", bound=BaseModel)

_NOT_RUNNING = "Ollama is not reachable at {url}. Start it with: ollama serve"


class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _body(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                *(m.model_dump() for m in messages),
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        body = self._body(system, messages, max_tokens, temperature) | {"stream": False}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(_NOT_RUNNING.format(url=self.base_url)) from exc
        except httpx.HTTPStatusError as exc:
            code, detail = exc.response.status_code, exc.response.text
            raise LLMError(f"Ollama error ({code}): {detail}") from exc

        return str(response.json()["message"]["content"]).strip()

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        body = self._body(system, messages, max_tokens, temperature) | {"stream": True}
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=body) as resp,
            ):
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content"):
                        yield content
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(_NOT_RUNNING.format(url=self.base_url)) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama error ({exc.response.status_code})") from exc

    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        schema: type[T],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> T:
        """Schema-constrained decoding — Ollama will not emit invalid JSON."""
        body = self._body(system, messages, max_tokens, temperature) | {
            "stream": False,
            "format": schema.model_json_schema(),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(_NOT_RUNNING.format(url=self.base_url)) from exc

        content = response.json()["message"]["content"]
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise LLMError(f"Ollama returned invalid {schema.__name__}: {exc}") from exc

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                tags = {m["name"] for m in response.json().get("models", [])}
        except Exception:
            return False
        # Ollama reports tags like "qwen2.5:7b-instruct"; accept a bare family name too.
        family = self.model.split(":")[0]
        return any(tag == self.model or tag.startswith(f"{family}:") for tag in tags)
