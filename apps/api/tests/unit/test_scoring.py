"""PyTorch model mechanics and the blending logic in the scoring service."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from app.scoring import service
from app.scoring.features import FEATURE_NAMES, N_FEATURES
from app.scoring.model import (
    N_TARGETS,
    TARGET_NAMES,
    AnswerScorer,
    ScorerCheckpoint,
    load_scorer,
    predict,
)


@pytest.fixture(autouse=True)
def no_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests offline — never download the embedding model."""
    monkeypatch.setattr(service.settings, "embeddings_enabled", False)


def test_forward_shape_and_range() -> None:
    model = AnswerScorer().eval()
    out = model(torch.randn(8, N_FEATURES) * 50)
    assert out.shape == (8, N_TARGETS)
    assert torch.all(out >= 0) and torch.all(out <= 100)


def test_zero_variance_feature_does_not_produce_nan() -> None:
    model = AnswerScorer()
    std = np.ones(N_FEATURES)
    std[0] = 0.0  # e.g. has_audio in a text-only corpus
    model.set_normalisation(np.zeros(N_FEATURES), std)
    assert torch.isfinite(model.eval()(torch.ones(2, N_FEATURES))).all()


def test_checkpoint_roundtrip_preserves_predictions(tmp_path: Path) -> None:
    model = AnswerScorer()
    model.set_normalisation(np.full(N_FEATURES, 3.0), np.full(N_FEATURES, 2.0))
    model.eval()
    x = np.random.default_rng(0).normal(size=(4, N_FEATURES)).astype(np.float32)
    before = predict(model, x)

    path = tmp_path / "scorer.pt"
    ScorerCheckpoint(state_dict=model.state_dict(), metrics={"r2_mean": 0.5}).save(path)
    loaded, checkpoint = load_scorer(path)

    np.testing.assert_allclose(predict(loaded, x), before, rtol=1e-5)
    assert checkpoint.metrics["r2_mean"] == 0.5
    # Normalisation stats travel inside the checkpoint.
    assert torch.allclose(loaded.feature_mean, torch.full((N_FEATURES,), 3.0))


def test_checkpoint_with_drifted_features_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    ScorerCheckpoint(
        state_dict=AnswerScorer().state_dict(),
        feature_names=(*FEATURE_NAMES[:-1], "renamed_feature"),
    ).save(path)
    with pytest.raises(ValueError, match="feature layout"):
        load_scorer(path)


def test_predict_accepts_single_vector() -> None:
    out = predict(AnswerScorer().eval(), np.zeros(N_FEATURES, dtype=np.float32))
    assert out.shape == (1, len(TARGET_NAMES))


@pytest.mark.parametrize(
    ("relevance", "expected"),
    [(None, 1.0), (0.9, 1.0), (0.45, 1.0), (0.05, 0.35), (0.15, 0.35)],
)
def test_relevance_multiplier_bounds(relevance: float | None, expected: float) -> None:
    assert service.relevance_multiplier(relevance) == pytest.approx(expected)


def test_relevance_multiplier_is_monotonic() -> None:
    values = [service.relevance_multiplier(r) for r in np.linspace(0, 1, 50)]
    assert all(b >= a for a, b in pairwise(values))


async def test_empty_answer_scores_zero() -> None:
    result = await service.score_answer(question="Tell me about yourself.", answer="   ")
    assert result.blended_score == 0.0


async def test_llm_scores_dominate_the_blend() -> None:
    answer = "I built a queue-backed retry layer that cut timeouts by 94 percent."
    high = await service.score_answer(question="q", answer=answer, llm_scores={"overall": 95.0})
    low = await service.score_answer(question="q", answer=answer, llm_scores={"overall": 10.0})
    assert high.blended_score > low.blended_score + 40


async def test_llm_dimensions_override_model_dimensions() -> None:
    result = await service.score_answer(
        question="q",
        answer="I led the migration and we shipped it in six weeks.",
        llm_scores={
            "structure": 11.0,
            "specificity": 22.0,
            "clarity": 33.0,
            "depth": 44.0,
            "overall": 50.0,
        },
    )
    assert (result.structure, result.specificity, result.clarity, result.depth) == (
        11.0,
        22.0,
        33.0,
        44.0,
    )


async def test_scoring_works_without_llm_scores() -> None:
    result = await service.score_answer(
        question="q", answer="I designed and built the ingestion service in Kafka."
    )
    assert result.llm_score is None
    assert 0.0 <= result.blended_score <= 100.0
    assert result.blended_score == pytest.approx(result.model_score)


def test_heuristic_fallback_ranks_strong_above_weak() -> None:
    from app.scoring.features import extract_features

    strong = service._heuristic_scores(
        extract_features(
            "When I was at my last company I designed a retry layer in Redis. As a result "
            "timeouts dropped by 94 percent across 300k daily requests, and I capped backoff "
            "at 3 seconds because retries added tail latency."
        )
    )
    weak = service._heuristic_scores(extract_features("Um I guess it was fine, you know."))
    assert strong["overall"] > weak["overall"]
