"""Interview lifecycle: create, answer, inspect, abandon."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import CurrentUser, DbSession
from app.models.enums import Competency, TurnKind
from app.schemas.interview import (
    AnswerResult,
    AnswerSubmit,
    InterviewCreate,
    InterviewOut,
    InterviewSummary,
    NextQuestionOut,
    ReportOut,
    ScoreOut,
)
from app.services import interview_service as svc

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/interviews", tags=["interviews"])


def _next_question(interview_id: uuid.UUID, step: svc.GraphStep) -> NextQuestionOut | None:
    if step.finished or not step.question:
        return None
    return NextQuestionOut(
        interview_id=interview_id,
        question=step.question,
        competency=step.competency,
        kind=step.kind,
        turn_index=step.turn_index,
        finished=False,
    )


@router.post("", response_model=NextQuestionOut, status_code=status.HTTP_201_CREATED)
async def create_interview(
    payload: InterviewCreate, user: CurrentUser, db: DbSession
) -> NextQuestionOut:
    """Start an interview and return its opening question."""
    try:
        interview, step = await svc.create_interview(
            db,
            user,
            role=payload.role,
            seniority=payload.seniority,
            persona=payload.persona,
            target_questions=payload.target_questions,
            resume_id=payload.resume_id,
        )
    except svc.InterviewError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return NextQuestionOut(
        interview_id=interview.id,
        question=step.question or "",
        competency=step.competency,
        kind=step.kind,
        turn_index=step.turn_index,
        opening_remark=(step.state or {}).get("opening_remark"),
    )


@router.get("", response_model=list[InterviewSummary])
async def list_interviews(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[InterviewSummary]:
    rows = await svc.list_interviews(db, user, limit=limit, offset=offset)
    return [InterviewSummary.model_validate(row) for row in rows]


@router.get("/{interview_id}", response_model=InterviewOut)
async def get_interview(interview_id: uuid.UUID, user: CurrentUser, db: DbSession) -> InterviewOut:
    try:
        interview = await svc.get_interview(db, user, interview_id)
    except svc.InterviewError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return InterviewOut.model_validate(interview)


@router.get("/{interview_id}/current", response_model=NextQuestionOut)
async def get_current_question(
    interview_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> NextQuestionOut:
    """The question awaiting an answer — used to resume after a page reload."""
    try:
        interview = await svc.get_interview(db, user, interview_id)
    except svc.InterviewError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    turn = await svc.current_question(db, interview)
    if turn is None:
        return NextQuestionOut(
            interview_id=interview.id,
            question="",
            competency=Competency.communication,
            kind=TurnKind.closing,
            turn_index=0,
            finished=True,
        )

    return NextQuestionOut(
        interview_id=interview.id,
        question=turn.question,
        competency=turn.competency,
        kind=turn.kind,
        turn_index=turn.index,
    )


@router.post("/{interview_id}/answer", response_model=AnswerResult)
async def submit_answer(
    interview_id: uuid.UUID, payload: AnswerSubmit, user: CurrentUser, db: DbSession
) -> AnswerResult:
    """Submit a typed answer and receive its score plus the next question."""
    try:
        interview = await svc.get_interview(db, user, interview_id)
        score, step = await svc.submit_answer(db, interview, answer=payload.answer)
    except svc.InterviewError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    report = None
    if step.finished:
        await db.refresh(interview)
        if interview.report is not None:
            report = ReportOut.model_validate(interview.report)

    return AnswerResult(
        score=ScoreOut.model_validate(score),
        next=_next_question(interview_id, step),
        report=report,
        finished=step.finished,
    )


@router.post("/{interview_id}/abandon", response_model=InterviewOut)
async def abandon_interview(
    interview_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> InterviewOut:
    try:
        interview = await svc.get_interview(db, user, interview_id)
    except svc.InterviewError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    interview = await svc.abandon_interview(db, interview)
    return InterviewOut.model_validate(interview)
