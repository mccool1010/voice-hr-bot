"""Speech-to-text via faster-whisper.

Replaces the browser's Web Speech API, which only works in Chrome and Edge,
gives no timing information, and silently returns nothing elsewhere. Running
Whisper server-side means every browser works, accented speech transcribes far
better, and — the part that matters for scoring — we get per-word offsets, which
`features.prosody` turns into pace, pause and hesitation signals.

The model is loaded once and reused. Transcription is CPU/GPU-bound, so every
call is dispatched to a worker thread.
"""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from app.config import SpeechProviderName, settings
from app.speech import groq_whisper

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = structlog.get_logger(__name__)

_model: WhisperModel | None = None
_device: str | None = None
_lock = threading.Lock()
_load_failed = False
_force_cpu = False


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float
    confidence: float | None = None
    words: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _resolve_device() -> tuple[str, str]:
    """Pick the best available device and a compute type that suits it.

    float16 on GPU is roughly 2x faster than float32 with no meaningful accuracy
    loss for speech. int8 on CPU is the only setting that keeps a base model
    near real time.
    """
    device = settings.whisper_device
    compute = settings.whisper_compute_type

    if _force_cpu:
        return "cpu", "int8"

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute == "default":
        compute = "float16" if device == "cuda" else "int8"

    return device, compute


def _load() -> WhisperModel | None:
    global _model, _device, _load_failed

    if _model is not None or _load_failed:
        return _model

    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            log.warning("whisper.unavailable", error=str(exc))
            _load_failed = True
            return None

        device, compute = _resolve_device()
        try:
            log.info(
                "whisper.loading", model=settings.whisper_model, device=device, compute_type=compute
            )
            _model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
            _device = device
            log.info("whisper.loaded", model=settings.whisper_model, device=device)
        except Exception as exc:
            if device == "cuda":
                # PyTorch seeing the GPU does not mean CTranslate2 can use it:
                # it needs its own CUDA/cuDNN libraries. CPU still works.
                log.warning("whisper.gpu_unavailable_falling_back_to_cpu", error=str(exc))
                _model = _load_cpu()
            else:
                log.warning("whisper.unavailable", error=str(exc))
            _load_failed = _model is None
    return _model


def _load_cpu() -> WhisperModel | None:
    """Load on CPU. Called with the lock held."""
    global _device, _force_cpu
    from faster_whisper import WhisperModel

    _force_cpu = True
    try:
        model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
    except Exception as exc:
        log.warning("whisper.unavailable", error=str(exc))
        return None
    _device = "cpu"
    log.info("whisper.loaded", model=settings.whisper_model, device="cpu")
    return model


def device() -> str | None:
    """Where Whisper is running: "groq", "cuda", "cpu", or None before loading."""
    return "groq" if _hosted() else _device


def _hosted() -> bool:
    return settings.speech_provider == SpeechProviderName.groq


def is_available() -> bool:
    if not settings.speech_enabled:
        return False
    if _hosted():
        return groq_whisper.is_configured()
    return not _load_failed


def _run(model: WhisperModel, audio_path: str, language: str | None) -> Transcript:
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        # Trims leading/trailing silence without clipping quiet speech.
        vad_parameters={"min_silence_duration_ms": 500},
    )

    words: list[dict[str, Any]] = []
    pieces: list[str] = []
    logprobs: list[float] = []

    # `segments` is a lazy generator: iterating it is what runs the model, so
    # this loop is where inference errors actually surface.
    for segment in segments:
        pieces.append(segment.text)
        logprobs.append(segment.avg_logprob)
        for word in segment.words or []:
            words.append(
                {
                    "word": word.word.strip(),
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "probability": round(float(word.probability), 4),
                }
            )

    # avg_logprob is a log probability; exponentiating gives a 0-1 figure that
    # reads sensibly as a confidence in the UI.
    confidence = None
    if logprobs:
        confidence = round(min(1.0, math.exp(sum(logprobs) / len(logprobs))), 4)

    return Transcript(
        text=" ".join(p.strip() for p in pieces).strip(),
        language=getattr(info, "language", language or "en"),
        duration_s=round(float(getattr(info, "duration", 0.0)), 3),
        confidence=confidence,
        words=words,
    )


def transcribe_sync(audio_path: str, *, language: str | None = "en") -> Transcript:
    """Transcribe a file on disk. Blocking — call through `transcribe`."""
    global _model

    model = _load()
    if model is None:
        raise TranscriptionError(
            "Speech recognition is unavailable on this server. Type your answer instead."
        )

    try:
        return _run(model, audio_path, language)
    except Exception as exc:
        if _device != "cuda":
            raise TranscriptionError(f"Could not transcribe the audio: {exc}") from exc
        # CTranslate2 loads cuBLAS/cuDNN lazily, so a missing GPU library only
        # shows up on the first real inference. Switch to CPU for good and retry.
        log.warning("whisper.gpu_inference_failed_falling_back_to_cpu", error=str(exc))
        with _lock:
            _model = _load_cpu()
        if _model is None:
            raise TranscriptionError(f"Could not transcribe the audio: {exc}") from exc

    try:
        return _run(_model, audio_path, language)
    except Exception as exc:
        raise TranscriptionError(f"Could not transcribe the audio: {exc}") from exc


async def transcribe(audio_path: str, *, language: str | None = "en") -> Transcript:
    if _hosted():
        try:
            result = await groq_whisper.transcribe(audio_path, language=language)
        except groq_whisper.GroqSpeechError as exc:
            raise TranscriptionError(str(exc)) from exc
        return Transcript(**result)
    return await asyncio.to_thread(transcribe_sync, audio_path, language=language)


async def warmup() -> None:
    # Hosted speech has nothing to load.
    if settings.speech_enabled and not _hosted():
        await asyncio.to_thread(_load)


def reset() -> None:
    """Test hook."""
    global _model, _device, _load_failed, _force_cpu
    _model = None
    _device = None
    _load_failed = False
    _force_cpu = False
