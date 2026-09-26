"""HTTP request/response models for interviews.

Kept separate from the ORM models so the wire format can evolve without a
migration, and separate from `graph/contracts.py` so a prompt change cannot
silently alter the public API.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.models.enums import (
    Competency,
    InterviewStatus,
    Persona,
    Seniority,
    TurnKind,
)
from app.scoring.features import filler_breakdown


class InterviewCreate(BaseModel):
    role: str = Field(min_length=2, max_length=160, examples=["Data Scientist"])
    seniority: Seniority = Seniority.mid
    persona: Persona = Persona.neutral
    target_questions: int = Field(default=6, ge=3, le=10)
    resume_id: uuid.UUID | None = None


class AnswerSubmit(BaseModel):
    """A typed answer. Audio answers go through the upload endpoint instead."""

    answer: str = Field(min_length=1, max_length=20_000)


class ScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    blended_score: float
    model_score: float
    llm_score: float | None = None
    relevance: float | None = None
    structure: float | None = None
    specificity: float | None = None
    clarity: float | None = None
    depth: float | None = None
    scorer_version: str


class TurnOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    index: int
    kind: TurnKind
    competency: Competency
    question: str
    answer_text: str | None = None
    audio_duration_s: float | None = None
    transcription_confidence: float | None = None
    answered_at: datetime | None = None
    score: ScoreOut | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def filler_words(self) -> dict[str, int]:
        """Filler words heard in the answer, for display only; never part of a score."""
        return filler_breakdown(self.answer_text or "")


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    overall_score: float
    competency_scores: dict[str, float]
    delivery_metrics: dict[str, float | int | str | None]
    summary: str
    strengths: list[str]
    improvements: list[str]
    recommended_focus: str | None = None


class InterviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    seniority: Seniority
    persona: Persona
    status: InterviewStatus
    target_questions: int
    llm_provider: str
    llm_model: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    turns: list[TurnOut] = Field(default_factory=list)
    report: ReportOut | None = None


class InterviewSummary(BaseModel):
    """Row shape for the history list — no transcript, so the list stays light."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    seniority: Seniority
    persona: Persona
    status: InterviewStatus
    created_at: datetime
    completed_at: datetime | None = None
    answered_questions: int = 0
    overall_score: float | None = None


class NextQuestionOut(BaseModel):
    """What the client needs to render the current prompt."""

    interview_id: uuid.UUID
    question: str
    competency: Competency
    kind: TurnKind
    turn_index: int
    finished: bool = False
    opening_remark: str | None = None


class AnswerResult(BaseModel):
    """Response to submitting an answer: the score, then whatever comes next."""

    score: ScoreOut
    next: NextQuestionOut | None = None
    report: ReportOut | None = None
    finished: bool = False


class TranscriptOut(BaseModel):
    text: str
    language: str
    duration_s: float
    confidence: float | None = None
    word_count: int = 0
