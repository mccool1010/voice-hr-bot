"""Hosted speech-to-text via Groq's Whisper API.

Used by the lean deploy image, which cannot fit a local Whisper model in the
512 MB of RAM a free host provides. Groq serves whisper-large-v3-turbo and, with
`verbose_json`, returns the same per-word timings local faster-whisper does —
so the pace, pause and hesitation features work identically either way.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import structlog

from app.config import settings

log = structlog.get_logger(__name__)


class GroqSpeechError(RuntimeError):
    """Raised when the hosted transcription call fails."""


def is_configured() -> bool:
    return bool(settings.groq_api_key)


async def transcribe(audio_path: str, *, language: str | None = "en") -> dict[str, Any]:
    """Return a dict with text, language, duration_s, confidence and words."""
    try:
        import groq
    except ImportError as exc:  # pragma: no cover
        raise GroqSpeechError("groq package is not installed") from exc

    if not settings.groq_api_key:
        raise GroqSpeechError("GROQ_API_KEY is not set")

    client = groq.AsyncGroq(api_key=settings.groq_api_key, max_retries=2)
    path = Path(audio_path)
    audio = await asyncio.to_thread(path.read_bytes)
    try:
        response = await client.audio.transcriptions.create(
            file=(path.name, audio),
            model=settings.groq_whisper_model,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=language or "en",
        )
    except groq.RateLimitError as exc:
        raise GroqSpeechError("Speech service is busy — please try again in a moment.") from exc
    except groq.APIConnectionError as exc:
        raise GroqSpeechError("Could not reach the speech service.") from exc
    except groq.APIStatusError as exc:
        raise GroqSpeechError(f"Speech service error ({exc.status_code}).") from exc

    return parse_response(response.to_dict(), fallback_language=language)


def parse_response(data: dict[str, Any], *, fallback_language: str | None) -> dict[str, Any]:
    """Normalise Groq's verbose_json into the shape the scorer expects."""
    words = [
        {
            "word": str(w.get("word", "")).strip(),
            "start": round(float(w["start"]), 3),
            "end": round(float(w["end"]), 3),
        }
        for w in data.get("words") or []
        if "start" in w and "end" in w
    ]

    # Same confidence definition as the local path: exp(mean segment log-prob).
    logprobs = [float(s["avg_logprob"]) for s in data.get("segments") or [] if "avg_logprob" in s]
    confidence = round(min(1.0, math.exp(sum(logprobs) / len(logprobs))), 4) if logprobs else None

    return {
        "text": str(data.get("text", "")).strip(),
        "language": data.get("language") or fallback_language or "en",
        "duration_s": round(float(data.get("duration") or 0.0), 3),
        "confidence": confidence,
        "words": words,
    }
