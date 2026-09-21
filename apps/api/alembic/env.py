"""Alembic environment.

The URL comes from application settings rather than alembic.ini, so migrations
and the app can never point at different databases. Autogenerate sees every
table because `app.models` imports them all onto `Base.metadata`.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.config import settings
from app.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# LangGraph's Postgres checkpointer creates and migrates its own tables. Without
# this filter, autogenerate treats them as orphans and emits DROP TABLE for
# them — which would silently wipe every in-progress interview.
EXTERNALLY_MANAGED_PREFIXES = ("checkpoint",)


def include_object(obj, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    if type_ == "table" and name and name.startswith(EXTERNALLY_MANAGED_PREFIXES):
        return False
    if type_ == "index" and reflected and compare_to is None:
        table = getattr(getattr(obj, "table", None), "name", "") or ""
        if table.startswith(EXTERNALLY_MANAGED_PREFIXES):
            return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
