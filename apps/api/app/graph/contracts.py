"""Structured outputs the LLM must produce.

These are deliberately separate from the HTTP schemas in `app/schemas`. They are
a contract with the model, so every field carries a description — the
descriptions become the JSON Schema the provider constrains decoding against,
and they do more prompting work than the system prompt does.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.enums import Competency


class PlannedTopic(BaseModel):
    """One area the interview intends to cover."""

    title: str = Field(description="Short topic label, 2-6 words.")
    competency: Competency = Field(description="Which competency this topic probes.")
    rationale: str = Field(
        description="One sentence on why this matters for the role and candidate."
    )
    opening_question: str = Field(
        description="The first question for this topic. One question only, no preamble."
    )


class InterviewPlan(BaseModel):
    """The interviewer's agenda, produced once at the start."""

    topics: list[PlannedTopic] = Field(
        description="Ordered topics to cover. Produce exactly the number requested.",
        min_length=1,
        max_length=10,
    )
    opening_remark: str = Field(
        description="A brief, warm greeting that names the role. Two sentences maximum."
    )


class NextQuestion(BaseModel):
    """A single interviewer utterance."""

    question: str = Field(
        description=(
            "The exact words the interviewer says. One question. No numbering, "
            "no 'Question 3:' prefix, no restating the candidate's answer at length."
        )
    )
    competency: Competency = Field(description="Competency this question probes.")


class AnswerEvaluation(BaseModel):
    """Rubric grading of one answer. All scores are 0-100."""

    structure: float = Field(
        ge=0,
        le=100,
        description="Is there a clear arc — situation, action, result? Rambling scores low.",
    )
    specificity: float = Field(
        ge=0,
        le=100,
        description="Concrete details, numbers, named tools. Vague generalities score low.",
    )
    clarity: float = Field(
        ge=0, le=100, description="Easy to follow on first hearing. Jargon soup scores low."
    )
    depth: float = Field(
        ge=0,
        le=100,
        description="Genuine understanding versus surface recall. Textbook definitions score low.",
    )
    overall: float = Field(ge=0, le=100, description="Holistic score for this answer.")
    notes: str = Field(
        description="Two sentences of private assessment. Not shown to the candidate mid-interview."
    )
    should_probe: bool = Field(
        description=(
            "True when a follow-up on this same topic would reveal more — the answer "
            "was promising but thin, or made a claim worth testing. False when the "
            "topic is exhausted or the answer was thorough."
        )
    )
    probe_focus: str | None = Field(
        default=None,
        description="If should_probe, the specific thing to dig into. Otherwise null.",
    )


class ReportDraft(BaseModel):
    """End-of-interview feedback written for the candidate."""

    summary: str = Field(
        description="Three to four sentences addressed to the candidate, second person."
    )
    strengths: list[str] = Field(
        description="Two to four specific strengths, each citing something they actually said.",
        min_length=1,
        max_length=5,
    )
    improvements: list[str] = Field(
        description="Two to four actionable improvements. Specific and kind, never generic.",
        min_length=1,
        max_length=5,
    )
    recommended_focus: str = Field(
        description="The single highest-leverage thing to practise before a real interview."
    )


class ResumeFacts(BaseModel):
    """Structured facts pulled from an uploaded CV."""

    headline: str = Field(description="One line describing the candidate, e.g. their current role.")
    years_experience: float = Field(
        ge=0, le=60, description="Best estimate of total professional years. 0 if a student."
    )
    skills: list[str] = Field(
        description="Concrete technical skills and tools named in the CV.", max_length=40
    )
    projects: list[str] = Field(
        description="Short titles of notable projects or achievements.", max_length=15
    )
    domains: list[str] = Field(
        description="Industries or problem domains worked in.", max_length=10
    )
