"""Shared test fixtures.

The suite runs with no external services: the LLM is `EchoProvider`, the graph
uses an in-memory checkpointer, and the database is a per-test SQLite file.

SQLite stands in for PostgreSQL because every table here uses portable types
except JSONB, which is remapped below. Tests that genuinely exercise Postgres
behaviour are marked `integration` and run against a real server in CI.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.types import JSON

# JSONB has no SQLite implementation; render it as JSON there instead.
JSONB_compile_registered = False


def _register_sqlite_jsonb() -> None:
    global JSONB_compile_registered
    if JSONB_compile_registered:
        return
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb(type_, compiler, **kw):  # type: ignore[no-untyped-def]
        return compiler.visit_JSON(JSON(), **kw)

    JSONB_compile_registered = True


_register_sqlite_jsonb()

from app.db.base import Base
from app.db.session import get_db
from app.llm import EchoProvider, set_provider
from app.models import User


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def engine(tmp_path) -> AsyncIterator:  # type: ignore[no-untyped-def]
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    try:
        engine = create_async_engine(url, echo=False)
    except Exception:
        pytest.skip("aiosqlite is not installed")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        # SQLite ignores foreign keys unless asked; without this, cascade
        # behaviour would differ from PostgreSQL and tests would pass falsely.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db(engine) -> AsyncIterator[AsyncSession]:  # type: ignore[no-untyped-def]
    maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session


@pytest.fixture
def echo_llm() -> Iterator[EchoProvider]:
    """Install a deterministic provider for the duration of a test."""
    provider = EchoProvider()
    set_provider(provider)
    yield provider
    set_provider(None)


@pytest.fixture
async def app_client(db, echo_llm) -> AsyncIterator[AsyncClient]:  # type: ignore[no-untyped-def]
    """HTTP client wired to the test database.

    `lifespan` is not run — warming Whisper and the embedding model would add
    tens of seconds per test for no benefit.
    """
    from app.main import app

    async def _override_db() -> AsyncIterator[AsyncSession]:
        yield db

    app.dependency_overrides[get_db] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
async def user(db) -> User:  # type: ignore[no-untyped-def]
    from app.core.security import hash_password

    record = User(
        email="candidate@example.com",
        hashed_password=hash_password("test-password-123"),
        display_name="Test Candidate",
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


@pytest.fixture
async def auth_headers(user) -> dict[str, str]:  # type: ignore[no-untyped-def]
    from app.core.security import create_access_token

    token, _ = create_access_token(user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def isolate_graph() -> Iterator[None]:
    """Give every test a fresh in-memory graph so threads cannot collide."""
    from langgraph.checkpoint.memory import InMemorySaver

    from app.graph.builder import build_graph, set_graph

    set_graph(build_graph(InMemorySaver()))
    yield
    set_graph(None)
