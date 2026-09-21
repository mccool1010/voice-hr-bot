"""Database URL handling for hosted Postgres providers."""

from __future__ import annotations

import pytest

from app.config import Environment, Settings

NEON = (
    "postgresql://neondb_owner:secret@ep-cool-lake-123.eu-central-1.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)


def _settings(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_neon_url_is_translated_for_asyncpg() -> None:
    url = _settings(DATABASE_URL=NEON).database_url
    assert url.startswith("postgresql+asyncpg://neondb_owner:secret@ep-cool-lake-123")
    assert "ssl=require" in url
    assert "sslmode" not in url
    assert "channel_binding" not in url


def test_neon_url_keeps_libpq_params_for_the_checkpointer() -> None:
    url = _settings(DATABASE_URL=NEON).checkpoint_database_url
    assert url == NEON


@pytest.mark.parametrize("scheme", ["postgres://", "postgresql://", "postgresql+asyncpg://"])
def test_any_postgres_scheme_is_accepted(scheme: str) -> None:
    settings = _settings(DATABASE_URL=f"{scheme}u:p@host:5432/db")
    assert settings.database_url == "postgresql+asyncpg://u:p@host:5432/db"
    assert settings.checkpoint_database_url == "postgresql://u:p@host:5432/db"


def test_url_is_built_from_parts_without_override() -> None:
    settings = _settings(postgres_host="db", postgres_password="pw")
    assert settings.database_url == "postgresql+asyncpg://voicehr:pw@db:5432/voicehr"
    assert settings.checkpoint_database_url == "postgresql://voicehr:pw@db:5432/voicehr"


def test_production_refuses_the_default_jwt_secret() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET"):
        _settings(environment=Environment.production)
    # A real secret is accepted.
    _settings(environment=Environment.production, jwt_secret="x" * 48)
