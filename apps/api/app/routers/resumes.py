"""Resume upload and listing."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlalchemy import select

from app.config import settings
from app.core.deps import CurrentUser, DbSession
from app.models.resume import Resume
from app.schemas.resume import ResumeFactsOut, ResumeOut
from app.services import resume_service as svc

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/resumes", tags=["resumes"])

EXCERPT_CHARS = 400


def _to_out(resume: Resume) -> ResumeOut:
    return ResumeOut(
        id=resume.id,
        filename=resume.filename,
        content_type=resume.content_type,
        size_bytes=resume.size_bytes,
        created_at=resume.created_at,
        extracted=ResumeFactsOut.model_validate(resume.extracted or {}),
        excerpt=resume.raw_text[:EXCERPT_CHARS],
    )


@router.post("", response_model=ResumeOut, status_code=status.HTTP_201_CREATED)
async def upload_resume(
    user: CurrentUser, db: DbSession, file: UploadFile = File(...)
) -> ResumeOut:
    """Upload a CV. Its content personalises subsequent interview questions."""
    data = await file.read()

    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    if len(data) > settings.max_resume_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File is too large. The limit is {settings.max_resume_bytes // 1024 // 1024} MB.",
        )

    try:
        kind = svc.detect_kind(file.content_type or "", file.filename or "")
        text = svc.normalise(svc.extract_text(data, kind))
    except svc.ResumeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if len(text) < 50:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Almost no text could be read from that file. If it is a scanned PDF, "
            "upload a text-based version instead.",
        )

    facts = await svc.extract_facts(text)

    resume = Resume(
        user_id=user.id,
        filename=file.filename or "resume",
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(data),
        raw_text=text,
        extracted=facts.model_dump(mode="json"),
    )
    db.add(resume)
    await db.commit()
    await db.refresh(resume)

    log.info("resume.uploaded", user_id=str(user.id), chars=len(text))
    return _to_out(resume)


@router.get("", response_model=list[ResumeOut])
async def list_resumes(user: CurrentUser, db: DbSession) -> list[ResumeOut]:
    rows = await db.scalars(
        select(Resume).where(Resume.user_id == user.id).order_by(Resume.created_at.desc())
    )
    return [_to_out(r) for r in rows.all()]


@router.delete("/{resume_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resume(resume_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    resume = await db.scalar(
        select(Resume).where(Resume.id == resume_id, Resume.user_id == user.id)
    )
    if resume is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found.")
    await db.delete(resume)
    await db.commit()
