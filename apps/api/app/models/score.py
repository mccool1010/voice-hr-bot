from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.interview import Interview, Turn


class TurnScore(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Hybrid score for one answer.

    Three independent signals are stored separately rather than collapsed, so the
    blend can be re-tuned later and so the PyTorch model can be evaluated against
    the LLM rubric as a weak label source:

      model_score  — PyTorch MLP over the engineered feature vector
      llm_score    — LLM grading against an explicit rubric
      relevance    — cosine similarity between question and answer embeddings

    `features` keeps the exact vector that produced `model_score`, which makes
    every score reproducible and gives the retraining job a labelled dataset.
    """

    __tablename__ = "turn_scores"

    turn_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("turns.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )

    model_score: Mapped[float] = mapped_column(Float, nullable=False)
    llm_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    blended_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    # Rubric dimensions, 0-100, produced by the LLM evaluator.
    structure: Mapped[float | None] = mapped_column(Float, nullable=True)
    specificity: Mapped[float | None] = mapped_column(Float, nullable=True)
    clarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    depth: Mapped[float | None] = mapped_column(Float, nullable=True)

    features: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict, nullable=False)
    rubric_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    scorer_version: Mapped[str] = mapped_column(String(32), nullable=False)

    turn: Mapped[Turn] = relationship(back_populates="score")

    def __repr__(self) -> str:
        return f"<TurnScore {self.blended_score:.1f}>"


class InterviewReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """End-of-interview summary. One per completed interview."""

    __tablename__ = "interview_reports"

    interview_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interviews.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )

    overall_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    competency_scores: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict, nullable=False)
    delivery_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    strengths: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    improvements: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    recommended_focus: Mapped[str | None] = mapped_column(Text, nullable=True)

    interview: Mapped[Interview] = relationship(back_populates="report")

    def __repr__(self) -> str:
        return f"<InterviewReport {self.overall_score:.1f}>"
