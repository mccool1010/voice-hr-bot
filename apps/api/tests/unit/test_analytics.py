"""Pandas analytics — pure functions, no database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.analytics import aggregate
from app.analytics.aggregate import ScoredTurn

START = datetime(2026, 9, 1, tzinfo=UTC)


def _turns(
    scores: list[float],
    competency: str = "communication",
    role: str = "SWE",
    features: dict[str, float] | None = None,
) -> list[ScoredTurn]:
    return [
        ScoredTurn(
            interview_id=f"i-{i // 3}",
            role=role,
            competency=competency,
            created_at=START + timedelta(hours=i),
            blended_score=s,
            structure=s,
            specificity=s,
            clarity=s,
            depth=s,
            features=features,
        )
        for i, s in enumerate(scores)
    ]


def test_empty_history_is_handled_everywhere() -> None:
    frame = aggregate.to_frame([])
    assert aggregate.overview(frame)["total_answers"] == 0
    assert aggregate.progress_series(frame) == []
    assert aggregate.competency_breakdown(frame) == []
    assert aggregate.delivery_metrics(frame) == {}
    assert aggregate.percentile_rank(frame, [50.0] * 20) is None


def test_overview_counts_and_improvement() -> None:
    frame = aggregate.to_frame(_turns([40, 45, 50, 55, 60, 65, 70, 75, 80]))
    stats = aggregate.overview(frame)

    assert stats["total_answers"] == 9
    assert stats["total_interviews"] == 3
    assert stats["best_score"] == 80
    assert stats["latest_score"] == 80
    # Last third (70,75,80) minus first third (40,45,50).
    assert stats["improvement"] == pytest.approx(30.0)


def test_improvement_needs_enough_history() -> None:
    stats = aggregate.overview(aggregate.to_frame(_turns([40, 90])))
    assert stats["improvement"] is None


def test_rolling_mean_smooths_the_series() -> None:
    series = aggregate.progress_series(aggregate.to_frame(_turns([0, 100, 0, 100, 0, 100])))
    assert [p["index"] for p in series] == [1, 2, 3, 4, 5, 6]
    assert series[0]["rolling"] == 0
    assert series[-1]["rolling"] == pytest.approx(60.0)  # mean of last five


def test_competency_breakdown_sorts_and_computes_delta() -> None:
    turns = _turns([40, 40, 80, 80], competency="ownership") + _turns([50], "collaboration")
    rows = aggregate.competency_breakdown(aggregate.to_frame(turns))

    assert [r["competency"] for r in rows] == ["ownership", "collaboration"]
    ownership = rows[0]
    assert ownership["answers"] == 4
    assert ownership["delta"] == pytest.approx(40.0)
    assert rows[1]["delta"] is None  # too few answers for a trend


def test_delivery_ignores_typed_answers() -> None:
    spoken = _turns([70], features={"words_per_minute": 140.0, "filler_rate": 2.0})
    typed = _turns([70], features={"words_per_minute": 0.0, "filler_rate": 9.0})
    metrics = aggregate.delivery_metrics(aggregate.to_frame(spoken + typed))

    assert metrics["words_per_minute"] == 140.0
    assert metrics["filler_rate"] == 2.0
    assert metrics["pace_verdict"] == "good"


@pytest.mark.parametrize(("wpm", "verdict"), [(90, "too slow"), (135, "good"), (190, "too fast")])
def test_pace_verdict(wpm: float, verdict: str) -> None:
    frame = aggregate.to_frame(_turns([50], features={"words_per_minute": wpm}))
    assert aggregate.delivery_metrics(frame)["pace_verdict"] == verdict


def test_percentile_rank() -> None:
    frame = aggregate.to_frame(_turns([75.0]))
    cohort = [float(x) for x in range(0, 100, 5)]  # 0..95; 15 of 20 fall below 75
    assert aggregate.percentile_rank(frame, cohort) == pytest.approx(75.0)


def test_percentile_requires_a_real_cohort() -> None:
    frame = aggregate.to_frame(_turns([75.0]))
    assert aggregate.percentile_rank(frame, [50.0] * 5) is None
