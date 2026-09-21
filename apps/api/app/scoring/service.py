"""Scoring orchestration: features → PyTorch → embeddings → blended score.

Three signals are combined rather than trusting any one:

  llm_score    content quality judged against a rubric. Strongest signal for
               *what* was said, but expensive and slightly non-deterministic.
  model_score  the PyTorch MLP over engineered features. Captures structure and
               delivery, is deterministic, costs ~0.2 ms, and still works when
               the LLM call fails.
  relevance    question-answer embedding similarity. Recorded and shown, but
               deliberately NOT part of the blend: ml/calibrate_relevance.py
               shows it cannot separate on-topic from off-topic answers in
               behavioural interviews (generic questions, specific stories).
               Off-topic answers are penalised by the rubric's anchors instead.

The weights below are the tuning surface. They live here as named constants
rather than being scattered through the graph.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.config import settings
from app.scoring import embeddings
from app.scoring.features import extract_features, features_to_vector
from app.scoring.model import (
    SCORER_VERSION,
    TARGET_NAMES,
    AnswerScorer,
    load_scorer,
    predict,
)

log = structlog.get_logger(__name__)

# Blend weights, applied to the content signals before the relevance gate.
W_LLM = 0.65
W_MODEL = 0.35


_model: AnswerScorer | None = None
_version: str = "untrained"
_load_attempted = False


@dataclass
class ScoreResult:
    """Everything the scorer produced for one answer."""

    blended_score: float
    model_score: float
    llm_score: float | None = None
    relevance: float | None = None
    structure: float | None = None
    specificity: float | None = None
    clarity: float | None = None
    depth: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    scorer_version: str = SCORER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_model() -> AnswerScorer | None:
    """Lazy-load the checkpoint. An untrained deployment degrades, it does not crash."""
    global _model, _version, _load_attempted

    if _model is not None or _load_attempted:
        return _model

    _load_attempted = True
    path = Path(settings.scorer_checkpoint)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path

    if not path.exists():
        log.warning(
            "scorer.checkpoint_missing",
            path=str(path),
            hint="run: python -m ml.train_scorer",
        )
        return None

    try:
        _model, checkpoint = load_scorer(path)
        _version = checkpoint.version
        log.info("scorer.loaded", version=_version, metrics=checkpoint.metrics)
    except Exception as exc:
        log.error("scorer.load_failed", error=str(exc))
        _model = None
    return _model


def score_features_only(
    answer: str,
    *,
    word_timings: list[dict[str, Any]] | None = None,
    audio_duration_s: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run feature extraction and the PyTorch head. No network calls.

    Returns (features, per-target predictions). When no checkpoint is present the
    predictions fall back to a transparent heuristic so the app stays usable.
    """
    features = extract_features(
        answer, word_timings=word_timings, audio_duration_s=audio_duration_s
    )
    model = _get_model()

    if model is None:
        return features, _heuristic_scores(features)

    vector = features_to_vector(features)
    predictions = predict(model, vector)[0]
    return features, {
        name: float(np.clip(value, 0.0, 100.0))
        for name, value in zip(TARGET_NAMES, predictions, strict=True)
    }


def _heuristic_scores(features: dict[str, float]) -> dict[str, float]:
    """Interpretable fallback when no trained checkpoint exists.

    Not a substitute for the model — it exists so a fresh clone without
    `ml/artifacts/` still returns sensible numbers instead of zeros.
    """
    words = features["word_count"]
    # Answers under ~25 words are almost never substantive; past ~150 the
    # returns flatten, so length credit saturates rather than growing forever.
    length = float(np.clip((words - 20.0) / 110.0, 0.0, 1.0))
    star = features["star_coverage"]
    detail = float(np.clip(features["number_density"] / 4.0, 0.0, 1.0))
    named = float(np.clip(features["named_entity_density"] / 6.0, 0.0, 1.0))
    diversity = float(np.clip(features["type_token_ratio"] * 1.6, 0.0, 1.0))
    noise = float(np.clip((features["hedge_rate"] + features["filler_rate"]) / 12.0, 0.0, 1.0))

    structure = 100.0 * (0.55 * star + 0.45 * length)
    specificity = 100.0 * (0.5 * detail + 0.3 * named + 0.2 * length)
    clarity = 100.0 * (0.6 * diversity + 0.4 * (1.0 - noise))
    depth = 100.0 * (0.4 * length + 0.35 * detail + 0.25 * star)
    overall = 0.3 * structure + 0.25 * specificity + 0.2 * clarity + 0.25 * depth

    return {
        "structure": round(structure, 2),
        "specificity": round(specificity, 2),
        "clarity": round(clarity, 2),
        "depth": round(depth, 2),
        "overall": round(overall, 2),
    }


async def score_answer(
    *,
    question: str,
    answer: str,
    llm_scores: dict[str, float] | None = None,
    word_timings: list[dict[str, Any]] | None = None,
    audio_duration_s: float | None = None,
) -> ScoreResult:
    """Full scoring pass for one answer.

    `llm_scores` comes from the graph's evaluate node. When it is absent — the
    evaluator errored, or scoring is running offline — the PyTorch prediction
    carries the whole content signal.
    """
    if not answer or not answer.strip():
        return ScoreResult(
            blended_score=0.0,
            model_score=0.0,
            relevance=0.0,
            features=extract_features(""),
            scorer_version=_version,
        )

    # Feature extraction and the forward pass are CPU-bound; keep them off the
    # event loop so concurrent interviews are not serialised behind each other.
    features, model_preds = await asyncio.to_thread(
        score_features_only,
        answer,
        word_timings=word_timings,
        audio_duration_s=audio_duration_s,
    )

    similarity = await embeddings.relevance(question, answer)

    model_score = model_preds["overall"]
    llm_score = llm_scores.get("overall") if llm_scores else None

    content = W_LLM * llm_score + W_MODEL * model_score if llm_score is not None else model_score

    blended = float(np.clip(content, 0.0, 100.0))

    # Prefer the LLM's per-dimension judgement where available; the model's
    # predictions fill in the rest.
    def dimension(name: str) -> float:
        if llm_scores and name in llm_scores:
            return float(llm_scores[name])
        return model_preds[name]

    return ScoreResult(
        blended_score=round(blended, 2),
        model_score=round(model_score, 2),
        llm_score=round(llm_score, 2) if llm_score is not None else None,
        relevance=round(similarity, 4) if similarity is not None else None,
        structure=round(dimension("structure"), 2),
        specificity=round(dimension("specificity"), 2),
        clarity=round(dimension("clarity"), 2),
        depth=round(dimension("depth"), 2),
        features=features,
        scorer_version=_version,
    )


def is_trained() -> bool:
    """True when a real checkpoint is loaded, rather than the heuristic fallback."""
    return _get_model() is not None


def version() -> str:
    """Scorer version string, or 'untrained' when running on the fallback."""
    _get_model()
    return _version


def warmup() -> None:
    """Load the checkpoint during startup rather than on first request."""
    _get_model()


def reset() -> None:
    """Test hook."""
    global _model, _load_attempted, _version
    _model = None
    _load_attempted = False
    _version = "untrained"
