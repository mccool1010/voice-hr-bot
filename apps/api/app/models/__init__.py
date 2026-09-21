"""ORM models. Importing this package registers every table on `Base.metadata`."""

from app.db.base import Base
from app.models.enums import (
    Competency,
    InterviewStatus,
    Persona,
    Seniority,
    TurnKind,
)
from app.models.interview import Interview, Turn
from app.models.resume import Resume
from app.models.score import InterviewReport, TurnScore
from app.models.user import User

__all__ = [
    "Base",
    "Competency",
    "Interview",
    "InterviewReport",
    "InterviewStatus",
    "Persona",
    "Resume",
    "Seniority",
    "Turn",
    "TurnKind",
    "TurnScore",
    "User",
]
