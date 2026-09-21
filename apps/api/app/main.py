"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.logging import configure_logging
from app.db.session import dispose_engine
from app.graph.builder import init_graph, shutdown_graph
from app.llm.base import LLMError, LLMUnavailableError
from app.llm.factory import get_provider
from app.routers import analytics, auth, health, interviews, resumes, ws
from app.scoring import embeddings
from app.scoring import service as scoring
from app.speech import transcribe

configure_logging()
log = structlog.get_logger(__name__)

DESCRIPTION = """
Practice job interviews with an AI interviewer that listens, adapts, and scores.

**How it works**

* A LangGraph state machine plans an agenda, asks questions, probes weak answers,
  and writes a closing report — with Postgres checkpointing, so an interview
  survives a dropped connection.
* Answers are transcribed server-side by Whisper, which yields per-word timings.
* Each answer is scored three ways: a PyTorch model over engineered features, an
  LLM rubric, and embedding relevance between question and answer.
* The LLM is pluggable — Ollama, Groq or Claude, selected by configuration.
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info(
        "app.starting",
        environment=settings.environment.value,
        provider=settings.llm_provider.value,
    )

    await init_graph()

    # Warm the models concurrently. Failures are logged and tolerated — the app
    # degrades to a smaller feature set rather than refusing to start.
    results = await asyncio.gather(
        embeddings.warmup(),
        transcribe.warmup(),
        asyncio.to_thread(scoring.warmup),
        return_exceptions=True,
    )
    for name, result in zip(("embeddings", "whisper", "scorer"), results, strict=True):
        if isinstance(result, Exception):
            log.warning("app.warmup_failed", component=name, error=str(result))

    try:
        provider = get_provider()
        log.info("app.ready", provider=provider.name, model=provider.model)
    except LLMUnavailableError as exc:
        log.error("app.no_llm_provider", error=str(exc))

    yield

    log.info("app.stopping")
    await shutdown_graph()
    await dispose_engine()


app = FastAPI(
    title="Voice HR — Interview Simulator API",
    description=DESCRIPTION,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    log.warning("api.llm_unavailable", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "code": "llm_unavailable"},
    )


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    log.warning("api.llm_error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc), "code": "llm_error"},
    )


for router in (health.router, auth.router, interviews.router, resumes.router, analytics.router):
    app.include_router(router, prefix=settings.api_v1_prefix)

# The WebSocket path is versioned in its own decorator, not prefixed.
app.include_router(ws.router)


def _mount_frontend(application: FastAPI) -> bool:
    """Serve the built SPA from the API when STATIC_DIR is configured.

    Used by the single-container deploy (Hugging Face Spaces, Railway). Unknown
    paths fall back to index.html so client-side routes survive a refresh.
    """
    static = settings.static_dir
    if static is None or not (static / "index.html").is_file():
        return False

    application.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")
    index = static / "index.html"

    @application.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        candidate = (static / path).resolve()
        if path and candidate.is_file() and static.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)

    return True


if not _mount_frontend(app):

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "Voice HR Interview Simulator",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
