"""Analytics over a candidate's interview history.

Everything here takes a flat list of scored turns and returns chart-ready
structures. Pandas does the work it is actually good at — time-indexed rolling
statistics, grouped aggregation, and quantile ranking — rather than being used
as a fancier list.

The functions are deliberately pure: routers query the database and hand rows
in, which makes every statistic testable without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Rolling window over recent answers, used for the trend line. Five is short
# enough to react within a single interview and long enough to damp one bad answer.
ROLLING_WINDOW = 5

# Feature columns surfaced as "delivery" metrics in the UI.
DELIVERY_FEATURES = (
    "words_per_minute",
    "filler_rate",
    "hedge_rate",
    "long_pause_rate",
    "silence_ratio",
    "star_coverage",
    "number_density",
)


@dataclass(frozen=True)
class ScoredTurn:
    """Flat row handed in by the router."""

    interview_id: str
    role: str
    competency: str
    created_at: datetime
    blended_score: float
    structure: float | None = None
    specificity: float | None = None
    clarity: float | None = None
    depth: float | None = None
    features: dict[str, float] | None = None


def to_frame(turns: list[ScoredTurn]) -> pd.DataFrame:
    """Build a tidy, time-sorted DataFrame with features flattened into columns."""
    if not turns:
        return pd.DataFrame(
            columns=[
                "interview_id",
                "role",
                "competency",
                "created_at",
                "blended_score",
                "structure",
                "specificity",
                "clarity",
                "depth",
            ]
        )

    base = pd.DataFrame(
        [
            {
                "interview_id": t.interview_id,
                "role": t.role,
                "competency": t.competency,
                "created_at": t.created_at,
                "blended_score": t.blended_score,
                "structure": t.structure,
                "specificity": t.specificity,
                "clarity": t.clarity,
                "depth": t.depth,
            }
            for t in turns
        ]
    )

    features = pd.DataFrame([t.features or {} for t in turns], index=base.index)
    keep = [c for c in DELIVERY_FEATURES if c in features.columns]

    frame = pd.concat([base, features[keep]], axis=1) if keep else base
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame.sort_values("created_at").reset_index(drop=True)


def overview(frame: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers for the dashboard."""
    if frame.empty:
        return {
            "total_interviews": 0,
            "total_answers": 0,
            "average_score": None,
            "best_score": None,
            "latest_score": None,
            "improvement": None,
            "roles_practised": [],
        }

    scores = frame["blended_score"]

    # Improvement compares the first and last thirds rather than first-vs-last
    # answer, which would be dominated by noise on a single response.
    third = max(1, len(scores) // 3)
    improvement: float | None = None
    if len(scores) >= 6:
        improvement = round(float(scores.iloc[-third:].mean() - scores.iloc[:third].mean()), 2)

    return {
        "total_interviews": int(frame["interview_id"].nunique()),
        "total_answers": len(frame),
        "average_score": round(float(scores.mean()), 2),
        "best_score": round(float(scores.max()), 2),
        "latest_score": round(float(scores.iloc[-1]), 2),
        "improvement": improvement,
        "roles_practised": sorted(frame["role"].dropna().unique().tolist()),
    }


def progress_series(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-answer scores with a rolling mean, ordered oldest to newest."""
    if frame.empty:
        return []

    series = frame[["created_at", "blended_score", "role"]].copy()
    series["rolling"] = (
        series["blended_score"].rolling(window=ROLLING_WINDOW, min_periods=1).mean().round(2)
    )
    series["index"] = range(1, len(series) + 1)

    return [
        {
            "index": int(row["index"]),
            "timestamp": row["created_at"].isoformat(),
            "score": round(float(row["blended_score"]), 2),
            "rolling": float(row["rolling"]),
            "role": row["role"],
        }
        for _, row in series.iterrows()
    ]


def competency_breakdown(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Mean score per competency, with a recent-versus-earlier delta."""
    if frame.empty:
        return []

    grouped = frame.groupby("competency")["blended_score"]
    summary = grouped.agg(["mean", "count", "std"]).reset_index()

    # Delta on the most recent half of each competency's answers.
    def recent_delta(group: pd.DataFrame) -> float | None:
        if len(group) < 4:
            return None
        half = len(group) // 2
        return round(
            float(
                group["blended_score"].iloc[half:].mean()
                - group["blended_score"].iloc[:half].mean()
            ),
            2,
        )

    deltas = {name: recent_delta(group) for name, group in frame.groupby("competency", sort=False)}

    return [
        {
            "competency": row["competency"],
            "score": round(float(row["mean"]), 2),
            "answers": int(row["count"]),
            "consistency": (
                round(float(100.0 - min(row["std"], 40.0) * 2.5), 1)
                if pd.notna(row["std"])
                else None
            ),
            "delta": deltas.get(row["competency"]),
        }
        for _, row in summary.sort_values("mean", ascending=False).iterrows()
    ]


def rubric_breakdown(frame: pd.DataFrame) -> dict[str, float | None]:
    """Mean of each rubric dimension — the radar chart's axes."""
    dimensions = ("structure", "specificity", "clarity", "depth")
    if frame.empty:
        return dict.fromkeys(dimensions, None)

    return {
        dim: (round(float(frame[dim].mean()), 2) if frame[dim].notna().any() else None)
        for dim in dimensions
        if dim in frame.columns
    }


def delivery_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Speaking-delivery statistics, averaged over answers that had audio."""
    available = [c for c in DELIVERY_FEATURES if c in frame.columns]
    if frame.empty or not available:
        return {}

    spoken = frame
    if "words_per_minute" in frame.columns:
        spoken = frame[frame["words_per_minute"] > 0]
    if spoken.empty:
        return {}

    metrics: dict[str, Any] = {col: round(float(spoken[col].mean()), 2) for col in available}

    wpm = metrics.get("words_per_minute")
    if wpm:
        # 110-160 wpm is the range listeners rate most highly for clarity.
        metrics["pace_verdict"] = "too slow" if wpm < 110 else "too fast" if wpm > 160 else "good"
    return metrics


def percentile_rank(frame: pd.DataFrame, cohort_scores: list[float]) -> float | None:
    """Where this candidate's mean score sits against everyone else's.

    Returns None below a cohort size where a percentile would be meaningless.
    """
    if frame.empty or len(cohort_scores) < 10:
        return None

    mine = float(frame["blended_score"].mean())
    cohort = np.asarray(cohort_scores, dtype=np.float64)
    return round(float((cohort < mine).mean() * 100.0), 1)


def build_report_delivery(features: list[dict[str, float]]) -> dict[str, Any]:
    """Condense one interview's per-answer features into report-level metrics."""
    if not features:
        return {}

    frame = pd.DataFrame(features)
    available = [c for c in DELIVERY_FEATURES if c in frame.columns]
    if not available:
        return {}

    spoken = frame[frame["words_per_minute"] > 0] if "words_per_minute" in frame else frame
    source = spoken if not spoken.empty else frame

    out: dict[str, Any] = {col: round(float(source[col].mean()), 2) for col in available}
    out["answers_analysed"] = len(frame)
    out["spoken_answers"] = len(spoken) if "words_per_minute" in frame else 0
    return out
