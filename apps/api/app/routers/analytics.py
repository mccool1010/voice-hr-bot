"""Analytics over a candidate's interview history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.analytics import aggregate
from app.core.deps import CurrentUser, DbSession
from app.models.enums import InterviewStatus
from app.models.interview import Interview, Turn
from app.models.score import TurnScore

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Below this many scored answers a cohort percentile is noise, not information.
MIN_COHORT = 10


async def _scored_turns(db: DbSession, user_id: Any, limit: int) -> list[aggregate.ScoredTurn]:
    rows = await db.execute(
        select(
            Interview.id,
            Interview.role,
            Turn.competency,
            TurnScore.created_at,
            TurnScore.blended_score,
            TurnScore.structure,
            TurnScore.specificity,
            TurnScore.clarity,
            TurnScore.depth,
            TurnScore.features,
        )
        .join(Turn, Turn.interview_id == Interview.id)
        .join(TurnScore, TurnScore.turn_id == Turn.id)
        .where(Interview.user_id == user_id)
        .order_by(TurnScore.created_at.asc())
        .limit(limit)
    )

    return [
        aggregate.ScoredTurn(
            interview_id=str(interview_id),
            role=role,
            competency=str(competency),
            created_at=created_at,
            blended_score=float(blended),
            structure=structure,
            specificity=specificity,
            clarity=clarity,
            depth=depth,
            features=features or {},
        )
        for (
            interview_id,
            role,
            competency,
            created_at,
            blended,
            structure,
            specificity,
            clarity,
            depth,
            features,
        ) in rows.all()
    ]


@router.get("/dashboard")
async def dashboard(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    """Everything the dashboard renders, in one round trip.

    One endpoint rather than five because every panel derives from the same
    DataFrame — splitting it would mean rebuilding that frame per request.
    """
    turns = await _scored_turns(db, user.id, limit)
    frame = aggregate.to_frame(turns)

    percentile = None
    if len(frame) >= 3:
        cohort = await db.scalars(
            select(TurnScore.blended_score)
            .join(Turn, Turn.id == TurnScore.turn_id)
            .join(Interview, Interview.id == Turn.interview_id)
            .where(Interview.user_id != user.id)
            .limit(5000)
        )
        scores = [float(s) for s in cohort.all()]
        if len(scores) >= MIN_COHORT:
            percentile = aggregate.percentile_rank(frame, scores)

    return {
        "overview": aggregate.overview(frame),
        "progress": aggregate.progress_series(frame),
        "competencies": aggregate.competency_breakdown(frame),
        "rubric": aggregate.rubric_breakdown(frame),
        "delivery": aggregate.delivery_metrics(frame),
        "percentile": percentile,
    }


@router.get("/summary")
async def summary(user: CurrentUser, db: DbSession) -> dict[str, Any]:
    """Lightweight counters for the navigation bar."""
    completed = await db.scalar(
        select(Interview)
        .where(Interview.user_id == user.id, Interview.status == InterviewStatus.completed)
        .with_only_columns(Interview.id)
        .limit(1)
    )
    turns = await _scored_turns(db, user.id, 500)
    frame = aggregate.to_frame(turns)
    return {
        "has_completed_interview": completed is not None,
        **aggregate.overview(frame),
    }
