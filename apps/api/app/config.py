"""Application settings, loaded from the environment."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# In the monorepo this is the repo root (apps/api/app/config.py → up three).
# Inside the container the tree is shallower, so fall back to the filesystem
# root; a missing .env there is simply ignored.
_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else _PARENTS[-1]


# libpq connection options that asyncpg does not understand.
_LIBPQ_ONLY_PARAMS = frozenset(
    {"channel_binding", "gssencmode", "target_session_attrs", "sslrootcert", "options"}
)


class Environment(StrEnum):
    development = "development"
    test = "test"
    production = "production"


class SpeechProviderName(StrEnum):
    """Where speech-to-text runs.

    local — faster-whisper in-process (CPU or GPU). Default; no network needed.
    groq  — Groq's hosted Whisper. For small deploy images with no local model.
    """

    local = "local"
    groq = "groq"


class LLMProviderName(StrEnum):
    """Which LLM backend serves interview turns.

    ollama    — local, GPU, no rate limits. Default for development.
    groq      — hosted, free tier, very fast. Default for the public demo.
    anthropic — hosted, highest quality reasoning. Used when a key is present.
    echo      — deterministic stub with no network. Used by the test suite.
    """

    ollama = "ollama"
    groq = "groq"
    anthropic = "anthropic"
    echo = "echo"


DEV_JWT_SECRET = "dev-only-insecure-jwt-secret-change-me-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── App ──────────────────────────────────────────────────────────────────
    environment: Environment = Environment.development
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173,http://localhost:4173"

    # ─── Database ─────────────────────────────────────────────────────────────
    postgres_user: str = "voicehr"
    postgres_password: str = "voicehr_dev_password"
    postgres_db: str = "voicehr"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url_override: str | None = Field(default=None, alias="DATABASE_URL")
    db_echo: bool = False

    # ─── Auth ─────────────────────────────────────────────────────────────────
    jwt_secret: str = DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60 * 24 * 30

    # ─── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: LLMProviderName = LLMProviderName.ollama
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: str = "high"

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    # Groq rate limits are per model, so on a 429 the provider fails over to
    # these immediately instead of sleeping on retry-after. Comma-separated.
    groq_fallback_models: str = "qwen/qwen3.8-27b,openai/gpt-oss-20b"
    # Only sent to gpt-oss models: "low" cuts reasoning tokens ~60% with
    # near-identical grades (measured), which matters on an 8k tokens/min free tier.
    groq_reasoning_effort: str = "low"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"

    # ─── Speech ───────────────────────────────────────────────────────────────
    speech_provider: SpeechProviderName = SpeechProviderName.local
    whisper_model: str = "base.en"
    groq_whisper_model: str = "whisper-large-v3-turbo"
    whisper_device: str = "auto"
    whisper_compute_type: str = "default"
    speech_enabled: bool = True
    max_audio_bytes: int = 25 * 1024 * 1024

    # ─── Scoring ──────────────────────────────────────────────────────────────
    scorer_checkpoint: Path = Path("ml/artifacts/answer_scorer.pt")
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    embeddings_enabled: bool = True

    # ─── Uploads ──────────────────────────────────────────────────────────────
    upload_dir: Path = Path("storage/uploads")

    # ─── Frontend ─────────────────────────────────────────────────────────────
    # When set, the API also serves the built SPA — one container for the demo.
    static_dir: Path | None = None
    max_resume_bytes: int = 5 * 1024 * 1024

    @field_validator("jwt_secret")
    @classmethod
    def _reject_default_secret_in_prod(cls, v: str, info) -> str:  # type: ignore[no-untyped-def]
        is_prod = info.data.get("environment") == Environment.production
        if v == DEV_JWT_SECRET and is_prod:
            raise ValueError("JWT_SECRET must be set to a real value in production")
        return v

    @property
    def _libpq_url(self) -> str:
        """The configured database as a plain libpq URL, query string intact."""
        if self.database_url_override:
            url = self.database_url_override
            for prefix in ("postgres://", "postgresql+asyncpg://", "postgresql+psycopg://"):
                if url.startswith(prefix):
                    return "postgresql://" + url[len(prefix) :]
            return url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Async SQLAlchemy DSN for asyncpg.

        Hosted Postgres (Neon, Supabase, Railway) hands out libpq URLs such as
        `...?sslmode=require&channel_binding=require`. asyncpg rejects libpq-only
        parameters, so they are translated: sslmode becomes asyncpg's `ssl`, and
        libpq connection options with no asyncpg equivalent are dropped.
        """
        parts = urlsplit(self._libpq_url)
        query: list[tuple[str, str]] = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            if key == "sslmode":
                query.append(("ssl", value))
            elif key not in _LIBPQ_ONLY_PARAMS:
                query.append((key, value))
        return urlunsplit(parts._replace(scheme="postgresql+asyncpg", query=urlencode(query)))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checkpoint_database_url(self) -> str:
        """Plain libpq URL for the LangGraph checkpointer (psycopg).

        Kept separate from `database_url` because psycopg needs the original
        libpq parameters — `sslmode` in particular — that asyncpg cannot take.
        """
        return self._libpq_url

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.production


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
