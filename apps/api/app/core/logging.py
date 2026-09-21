"""Structured logging.

JSON in production so log aggregators can parse it; coloured key-value pairs in
development so a human can read it.
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.config import Environment, settings


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    # Uvicorn's access log duplicates what the app already records.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # sentence-transformers and faster-whisper are chatty at INFO.
    for noisy in ("sentence_transformers", "faster_whisper", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.environment is Environment.production
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
