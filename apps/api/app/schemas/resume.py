"""Resume upload payloads."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ResumeFactsOut(BaseModel):
    headline: str = ""
    years_experience: float = 0.0
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class ResumeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
    extracted: ResumeFactsOut = Field(default_factory=ResumeFactsOut)
    excerpt: str = ""
