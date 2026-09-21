"""Semantic relevance between a question and an answer.

The engineered features in `features.py` measure *how* something was said. They
cannot tell whether it answered the question — a polished, well-structured reply
about the wrong topic scores highly on every lexical signal. Embedding cosine
similarity closes that gap.

The model is loaded lazily and cached process-wide. Loading takes a few seconds
and ~90 MB, so it must never happen inside a request path on a cold container;
`warmup()` is called from the application lifespan instead.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import numpy as np
import structlog

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger(__name__)

_model: SentenceTransformer | None = None
_lock = threading.Lock()
_load_failed = False


def _load() -> SentenceTransformer | None:
    """Load once, under a lock. Returns None if embeddings are unavailable."""
    global _model, _load_failed

    if _model is not None or _load_failed:
        return _model

    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from sentence_transformers import SentenceTransformer

            log.info("embeddings.loading", model=settings.embedding_model)
            _model = SentenceTransformer(settings.embedding_model)
            log.info("embeddings.loaded", model=settings.embedding_model)
        except Exception as exc:
            # A missing model download must degrade the score, not break the interview.
            log.warning("embeddings.unavailable", error=str(exc))
            _load_failed = True
    return _model


def is_available() -> bool:
    return settings.embeddings_enabled and not _load_failed


def relevance_sync(question: str, answer: str) -> float | None:
    """Cosine similarity in 0-1, or None when embeddings are unavailable.

    Similarity is clamped from [-1, 1] to [0, 1]; negative similarity between two
    English sentences carries no more meaning than zero for our purposes.
    """
    if not settings.embeddings_enabled:
        return None
    if not question.strip() or not answer.strip():
        return None

    model = _load()
    if model is None:
        return None

    try:
        vectors = model.encode(
            [question, answer], normalize_embeddings=True, show_progress_bar=False
        )
    except Exception as exc:  # pragma: no cover
        log.warning("embeddings.encode_failed", error=str(exc))
        return None

    similarity = float(np.dot(vectors[0], vectors[1]))
    return max(0.0, min(1.0, similarity))


async def relevance(question: str, answer: str) -> float | None:
    """Async wrapper — encoding is CPU-bound, so it runs off the event loop."""
    return await asyncio.to_thread(relevance_sync, question, answer)


async def warmup() -> None:
    """Preload during application startup so the first request is not slow."""
    if settings.embeddings_enabled:
        await asyncio.to_thread(_load)


def reset() -> None:
    """Test hook — drops the cached model and the failure latch."""
    global _model, _load_failed
    _model = None
    _load_failed = False
