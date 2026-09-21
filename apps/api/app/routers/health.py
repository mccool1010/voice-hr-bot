"""Health and capability reporting.

`/health` is the liveness probe — cheap and dependency-free, so a container
orchestrator can use it. `/health/ready` actually exercises the database and the
LLM provider, which is what you want a deploy gate to check.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.config import settings
from app.core.deps import DbSession
from app.llm.factory import get_provider
from app.scoring import embeddings
from app.scoring import service as scoring
from app.speech import transcribe

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment.value}


@router.get("/health/ready")
async def readiness(db: DbSession, response: Response) -> dict[str, Any]:
    """Deep check. Returns 503 if a hard dependency is down."""
    checks: dict[str, Any] = {}

    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        checks["database"] = {"ok": False, "error": str(exc)[:200]}

    try:
        provider = get_provider()
        checks["llm"] = {
            "ok": await provider.health(),
            "provider": provider.name,
            "model": provider.model,
        }
    except Exception as exc:
        checks["llm"] = {"ok": False, "error": str(exc)[:200]}

    # Degraded-but-serving: the app still works without these.
    checks["speech"] = {"ok": transcribe.is_available(), "model": settings.whisper_model}
    checks["embeddings"] = {"ok": embeddings.is_available(), "model": settings.embedding_model}
    checks["scorer"] = {"ok": scoring.is_trained(), "version": scoring.version()}

    hard_dependencies_ok = checks["database"]["ok"] and checks["llm"]["ok"]
    if not hard_dependencies_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": "ready" if hard_dependencies_ok else "degraded", "checks": checks}


@router.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    """What this deployment can do — the frontend adapts its UI to this."""
    provider = get_provider()
    return {
        "llm": {"provider": provider.name, "model": provider.model},
        "speech_to_text": transcribe.is_available(),
        "semantic_relevance": embeddings.is_available(),
        "trained_scorer": scoring.is_trained(),
        "scorer_version": scoring.version(),
        "max_audio_bytes": settings.max_audio_bytes,
        "max_resume_bytes": settings.max_resume_bytes,
    }
