"""Password hashing and JWT issuance."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.config import settings

# bcrypt only reads the first 72 bytes of a password. Truncating explicitly
# keeps behaviour identical across bcrypt versions (5.x raises instead).
_BCRYPT_MAX_BYTES = 72


def _secret(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


ALGORITHM = settings.jwt_algorithm


class TokenError(Exception):
    """Raised when a token is missing, malformed, or expired."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_secret(password), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_secret(plain), hashed.encode("ascii"))
    except ValueError:
        # Malformed hash in the database — treat as a failed login, not a 500.
        return False


def create_access_token(
    subject: uuid.UUID | str, *, extra_claims: dict[str, Any] | None = None
) -> tuple[str, int]:
    """Return (token, seconds_until_expiry)."""
    ttl = timedelta(minutes=settings.access_token_ttl_minutes)
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        "type": "access",
        **(extra_claims or {}),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)
    return token, int(ttl.total_seconds())


def decode_token(token: str) -> dict[str, Any]:
    try:
        payload: dict[str, Any] = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Your session has expired. Please sign in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid authentication token.") from exc

    if payload.get("type") != "access":
        raise TokenError("Invalid authentication token.")
    return payload


def subject_from_token(token: str) -> uuid.UUID:
    payload = decode_token(token)
    try:
        return uuid.UUID(str(payload["sub"]))
    except (KeyError, ValueError) as exc:
        raise TokenError("Invalid authentication token.") from exc
