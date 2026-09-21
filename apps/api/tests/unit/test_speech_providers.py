"""Speech provider selection and Groq Whisper response handling."""

from __future__ import annotations

from typing import Any

import pytest

from app.config import SpeechProviderName
from app.speech import groq_whisper, transcribe

GROQ_RESPONSE = {
    "text": " I built the retry layer in Redis.",
    "language": "English",
    "duration": 3.2,
    "segments": [{"avg_logprob": -0.1}, {"avg_logprob": -0.3}],
    "words": [
        {"word": "I", "start": 0.1, "end": 0.2},
        {"word": "built", "start": 0.25, "end": 0.6},
        {"word": "broken"},  # no timings — must be skipped, not crash
    ],
}


def test_parse_groq_response() -> None:
    parsed = groq_whisper.parse_response(GROQ_RESPONSE, fallback_language="en")
    assert parsed["text"] == "I built the retry layer in Redis."
    assert parsed["duration_s"] == 3.2
    assert [w["word"] for w in parsed["words"]] == ["I", "built"]
    assert parsed["confidence"] == pytest.approx(0.8187, abs=1e-4)  # exp(-0.2)


def test_parse_handles_missing_segments() -> None:
    parsed = groq_whisper.parse_response({"text": "hi"}, fallback_language="en")
    assert parsed["confidence"] is None
    assert parsed["words"] == []
    assert parsed["language"] == "en"


@pytest.fixture
def hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe.settings, "speech_provider", SpeechProviderName.groq)
    monkeypatch.setattr(transcribe.settings, "speech_enabled", True)


def test_hosted_availability_follows_the_api_key(
    hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transcribe.settings, "groq_api_key", None)
    assert not transcribe.is_available()
    monkeypatch.setattr(transcribe.settings, "groq_api_key", "gsk_test")
    assert transcribe.is_available()
    assert transcribe.device() == "groq"


async def test_hosted_transcription_is_dispatched_to_groq(
    hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(path: str, *, language: str | None = "en") -> dict[str, Any]:
        return groq_whisper.parse_response(GROQ_RESPONSE, fallback_language=language)

    monkeypatch.setattr(groq_whisper, "transcribe", fake)
    result = await transcribe.transcribe("answer.webm")
    assert result.text.startswith("I built")
    assert len(result.words) == 2


async def test_hosted_errors_become_transcription_errors(
    hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(path: str, *, language: str | None = "en") -> dict[str, Any]:
        raise groq_whisper.GroqSpeechError("Speech service is busy")

    monkeypatch.setattr(groq_whisper, "transcribe", failing)
    with pytest.raises(transcribe.TranscriptionError, match="busy"):
        await transcribe.transcribe("answer.webm")
