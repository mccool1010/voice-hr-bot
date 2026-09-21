"""Domain enums shared by ORM models, Pydantic schemas and the LangGraph state."""

from __future__ import annotations

from enum import StrEnum


class InterviewStatus(StrEnum):
    created = "created"
    in_progress = "in_progress"
    completed = "completed"
    abandoned = "abandoned"


class Seniority(StrEnum):
    intern = "intern"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    staff = "staff"


class Persona(StrEnum):
    """Interviewer style. Changes tone and follow-up aggressiveness, not fairness."""

    friendly = "friendly"
    neutral = "neutral"
    skeptical = "skeptical"
    time_pressed = "time_pressed"


class Competency(StrEnum):
    """What each question is probing. Scores roll up per competency."""

    communication = "communication"
    technical_depth = "technical_depth"
    problem_solving = "problem_solving"
    ownership = "ownership"
    collaboration = "collaboration"
    culture_fit = "culture_fit"


class TurnKind(StrEnum):
    opening = "opening"
    main = "main"
    follow_up = "follow_up"
    closing = "closing"
