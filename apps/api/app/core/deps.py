"""FastAPI dependencies."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Query, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import TokenError, subject_from_token
from app.db.session import get_db
from app.llm.base import BaseLLMProvider
from app.llm.factory import get_provider
from app.models.user import User

bearer_scheme = HTTPBearer(auto_error=False, description="JWT from /auth/login")

DbSession = Annotated[AsyncSession, Depends(get_db)]

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


async def _user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User:
    user = await db.scalar(select(User).where(User.id == user_id))
    if user is None or not user.is_active:
        raise _UNAUTHORIZED
    return user


async def current_user(
    db: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
) -> User:
    if credentials is None:
        raise _UNAUTHORIZED
    try:
        user_id = subject_from_token(credentials.credentials)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return await _user_by_id(db, user_id)


async def websocket_user(
    websocket: WebSocket,
    db: DbSession,
    token: Annotated[str | None, Query(description="JWT — browsers cannot set WS headers")] = None,
) -> User | None:
    """Authenticate a WebSocket.

    The browser WebSocket API cannot send an Authorization header, so the token
    arrives as a query parameter. The socket is closed with a policy-violation
    code rather than raising, since HTTP exceptions do not apply post-handshake.
    """
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
        return None
    try:
        user_id = subject_from_token(token)
        return await _user_by_id(db, user_id)
    except (TokenError, HTTPException):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        return None


def llm_provider() -> BaseLLMProvider:
    return get_provider()


CurrentUser = Annotated[User, Depends(current_user)]
LLM = Annotated[BaseLLMProvider, Depends(llm_provider)]
