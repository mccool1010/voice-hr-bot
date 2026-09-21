"""Whisper loading and the GPU → CPU fallback, with a fake model.

CTranslate2 can fail on the GPU either when the model loads or — because it
loads cuBLAS lazily — on the first inference. Both must fall back to CPU
rather than switching speech off.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from app.speech import transcribe

pytest.importorskip("faster_whisper")


def _segment(text: str) -> SimpleNamespace:
    words = [
        SimpleNamespace(word=w, start=i * 0.4, end=i * 0.4 + 0.3, probability=0.9)
        for i, w in enumerate(text.split())
    ]
    return SimpleNamespace(text=text, avg_logprob=-0.2, words=words)


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel."""

    fail_load_on: ClassVar[set[str]] = set()
    fail_inference_on: ClassVar[set[str]] = set()
    created: ClassVar[list[str]] = []

    def __init__(self, name: str, device: str, compute_type: str) -> None:
        FakeWhisper.created.append(device)
        if device in FakeWhisper.fail_load_on:
            raise RuntimeError(f"cannot load on {device}")
        self.device = device

    def transcribe(self, path: str, **_: Any) -> tuple[Any, SimpleNamespace]:
        info = SimpleNamespace(language="en", duration=2.0)

        def segments() -> Any:
            # Errors surface while iterating, exactly like the real generator.
            if self.device in FakeWhisper.fail_inference_on:
                raise RuntimeError("Library cublas64_12.dll is not found")
            yield _segment("I built the retry layer")

        return segments(), info


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> Any:
    import faster_whisper

    FakeWhisper.fail_load_on = set()
    FakeWhisper.fail_inference_on = set()
    FakeWhisper.created = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    monkeypatch.setattr(transcribe.settings, "speech_enabled", True)
    monkeypatch.setattr(transcribe.settings, "whisper_device", "cuda")
    transcribe.reset()
    yield FakeWhisper
    transcribe.reset()


def test_uses_gpu_when_it_works() -> None:
    result = transcribe.transcribe_sync("answer.webm")
    assert result.text == "I built the retry layer"
    assert len(result.words) == 5
    assert transcribe.device() == "cuda"


def test_gpu_load_failure_falls_back_to_cpu(fake_whisper: Any) -> None:
    fake_whisper.fail_load_on = {"cuda"}
    result = transcribe.transcribe_sync("answer.webm")
    assert result.text == "I built the retry layer"
    assert transcribe.device() == "cpu"
    assert transcribe.is_available()


def test_gpu_inference_failure_retries_on_cpu(fake_whisper: Any) -> None:
    """The lazy-cuBLAS case: load succeeds, the first inference does not."""
    fake_whisper.fail_inference_on = {"cuda"}
    result = transcribe.transcribe_sync("answer.webm")
    assert result.text == "I built the retry layer"
    assert fake_whisper.created == ["cuda", "cpu"]
    # Later calls go straight to CPU.
    transcribe.transcribe_sync("again.webm")
    assert fake_whisper.created == ["cuda", "cpu"]


def test_cpu_inference_failure_is_a_friendly_error(fake_whisper: Any) -> None:
    fake_whisper.fail_inference_on = {"cuda", "cpu"}
    with pytest.raises(transcribe.TranscriptionError, match="Could not transcribe"):
        transcribe.transcribe_sync("answer.webm")


def test_nothing_loads_means_speech_is_unavailable(fake_whisper: Any) -> None:
    fake_whisper.fail_load_on = {"cuda", "cpu"}
    with pytest.raises(transcribe.TranscriptionError, match="unavailable"):
        transcribe.transcribe_sync("answer.webm")
    assert not transcribe.is_available()
