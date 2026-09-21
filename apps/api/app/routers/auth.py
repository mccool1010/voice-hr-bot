"""Registration, login, and the demo account."""

from __future__ import annotations

import secrets

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.deps import CurrentUser, DbSession
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import TokenPair, UserLogin, UserOut, UserRegister

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

DEMO_EMAIL = "demo@voicehr.app"


def _token_response(user: User) -> TokenPair:
    token, expires_in = create_access_token(user.id, extra_claims={"email": user.email})
    return TokenPair(
        access_token=token,
        expires_in=expires_in,
        user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def register(payload: UserRegister, db: DbSession) -> TokenPair:
    user = User(
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
        display_name=payload.display_name.strip(),
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        # A unique-violation is the only realistic cause here, and reporting it
        # plainly is fine: this is a practice tool, not a system where account
        # enumeration carries meaningful risk.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        ) from None

    await db.refresh(user)
    log.info("auth.registered", user_id=str(user.id))
    return _token_response(user)


@router.post("/login", response_model=TokenPair)
async def login(payload: UserLogin, db: DbSession) -> TokenPair:
    user = await db.scalar(select(User).where(User.email == payload.email.lower()))

    # Always run the hash comparison, even when no user matched, so response
    # time does not reveal whether the address is registered.
    dummy_hash = "$2b$12$" + "." * 53
    valid = verify_password(payload.password, user.hashed_password if user else dummy_hash)

    if user is None or not valid or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    return _token_response(user)


@router.post("/demo", response_model=TokenPair)
async def demo_login(db: DbSession) -> TokenPair:
    """Sign in as the shared demo account, creating it on first use.

    The public demo needs a one-click path — asking a recruiter to register
    before they can see anything loses most of them.
    """
    user = await db.scalar(select(User).where(User.email == DEMO_EMAIL))

    if user is None:
        user = User(
            email=DEMO_EMAIL,
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            display_name="Demo Candidate",
            is_demo=True,
        )
        db.add(user)
        try:
            await db.commit()
        except IntegrityError:
            # Two first-time visitors can race here.
            await db.rollback()
            user = await db.scalar(select(User).where(User.email == DEMO_EMAIL))
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Could not start a demo session.",
                ) from None
        else:
            await db.refresh(user)

    return _token_response(user)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
