from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import Competency, InterviewStatus, Persona, Seniority, TurnKind

if TYPE_CHECKING:
    from app.models.score import InterviewReport, TurnScore
    from app.models.user import User


class Interview(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One interview session.

    `thread_id` is the LangGraph checkpoint thread. The graph owns conversational
    state; this table owns the queryable domain record that analytics reads.
    """

    __tablename__ = "interviews"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True
    )

    role: Mapped[str] = mapped_column(String(160), nullable=False)
    seniority: Mapped[Seniority] = mapped_column(
        SAEnum(Seniority, name="seniority"), default=Seniority.mid, nullable=False
    )
    persona: Mapped[Persona] = mapped_column(
        SAEnum(Persona, name="persona"), default=Persona.neutral, nullable=False
    )
    status: Mapped[InterviewStatus] = mapped_column(
        SAEnum(InterviewStatus, name="interview_status"),
        default=InterviewStatus.created,
        nullable=False,
        index=True,
    )

    thread_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    plan: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    target_questions: Mapped[int] = mapped_column(Integer, default=6, nullable=False)

    llm_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(120), nullable=False)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="interviews")
    turns: Mapped[list[Turn]] = relationship(
        back_populates="interview",
        cascade="all, delete-orphan",
        order_by="Turn.index",
        lazy="selectin",
    )
    report: Mapped[InterviewReport | None] = relationship(
        back_populates="interview", cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )

    __table_args__ = (Index("ix_interviews_user_status", "user_id", "status"),)

    def __repr__(self) -> str:
        return f"<Interview {self.role} [{self.status}]>"


class Turn(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single question/answer exchange.

    `word_timings` holds Whisper's per-word offsets, which the prosody feature
    extractor turns into pace, pause and hesitation signals. It is null when the
    answer arrived as typed text rather than audio.
    """

    __tablename__ = "turns"

    interview_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interviews.id", ondelete="CASCADE"), index=True, nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[TurnKind] = mapped_column(
        SAEnum(TurnKind, name="turn_kind"), default=TurnKind.main, nullable=False
    )
    competency: Mapped[Competency] = mapped_column(
        SAEnum(Competency, name="competency"), default=Competency.communication, nullable=False
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    audio_duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    transcription_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    word_timings: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    interview: Mapped[Interview] = relationship(back_populates="turns")
    score: Mapped[TurnScore | None] = relationship(
        back_populates="turn", cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("interview_id", "index", name="uq_turns_interview_id_index"),
    )

    def __repr__(self) -> str:
        return f"<Turn {self.index} of {self.interview_id}>"
