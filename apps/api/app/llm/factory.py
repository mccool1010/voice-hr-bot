"""Provider selection and graceful degradation.

Resolution order when `LLM_PROVIDER` is set is simply "use that one". The
interesting case is a misconfigured deployment: rather than 500 on every
request, the factory falls back down a preference chain and reports which
provider actually answered, so `/health` can surface it.
"""

from __future__ import annotations

import structlog

from app.config import LLMProviderName, Settings
from app.config import settings as default_settings
from app.llm.base import BaseLLMProvider, LLMUnavailableError
from app.llm.echo_provider import EchoProvider

log = structlog.get_logger(__name__)

_cached: BaseLLMProvider | None = None

# Tried in order when the configured provider cannot be constructed.
_FALLBACK_ORDER: tuple[LLMProviderName, ...] = (
    LLMProviderName.anthropic,
    LLMProviderName.groq,
    LLMProviderName.ollama,
)


def _construct(name: LLMProviderName, cfg: Settings) -> BaseLLMProvider:
    match name:
        case LLMProviderName.anthropic:
            from app.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                api_key=cfg.anthropic_api_key or "",
                model=cfg.anthropic_model,
                effort=cfg.anthropic_effort,
            )
        case LLMProviderName.groq:
            from app.llm.groq_provider import GroqProvider

            return GroqProvider(api_key=cfg.groq_api_key or "", model=cfg.groq_model)
        case LLMProviderName.ollama:
            from app.llm.ollama_provider import OllamaProvider

            return OllamaProvider(base_url=cfg.ollama_base_url, model=cfg.ollama_model)
        case LLMProviderName.echo:
            return EchoProvider()

    raise LLMUnavailableError(f"Unknown provider: {name}")  # pragma: no cover


def build_provider(cfg: Settings | None = None) -> BaseLLMProvider:
    """Construct the configured provider, falling back if it cannot be built.

    Construction failures are configuration problems (missing key, missing
    package) — not transient ones — so falling back here is safe. Network
    failures surface at call time instead, where the caller can retry.
    """
    cfg = cfg or default_settings

    try:
        provider = _construct(cfg.llm_provider, cfg)
        log.info("llm.provider.selected", provider=provider.name, model=provider.model)
        return provider
    except LLMUnavailableError as exc:
        log.warning("llm.provider.unavailable", provider=cfg.llm_provider, error=str(exc))

    for candidate in _FALLBACK_ORDER:
        if candidate == cfg.llm_provider:
            continue
        try:
            provider = _construct(candidate, cfg)
        except LLMUnavailableError:
            continue
        log.warning("llm.provider.fallback", requested=cfg.llm_provider, using=provider.name)
        return provider

    log.error("llm.provider.none_available")
    raise LLMUnavailableError(
        "No LLM provider could be configured. Set one of ANTHROPIC_API_KEY, "
        "GROQ_API_KEY, or run Ollama locally."
    )


def get_provider() -> BaseLLMProvider:
    """Process-wide singleton. FastAPI dependency and graph nodes both use this."""
    global _cached
    if _cached is None:
        _cached = build_provider()
    return _cached


def set_provider(provider: BaseLLMProvider | None) -> None:
    """Override the singleton. Tests inject `EchoProvider` through this."""
    global _cached
    _cached = provider
