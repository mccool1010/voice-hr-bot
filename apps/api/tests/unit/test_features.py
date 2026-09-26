"""Feature extraction must be stable, bounded, and actually discriminative."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.scoring.features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_batch,
    extract_features,
    features_to_vector,
    filler_breakdown,
)

STRONG_ANSWER = (
    "When I was at my last company we were running a payments service that kept "
    "timing out under load. I designed a queue-backed retry layer and I implemented "
    "it over two sprints using Redis and PostgreSQL. As a result timeouts dropped by "
    "94 percent and we stopped getting paged at night. The trade-off was that retries "
    "added tail latency, so I capped the backoff at 3 seconds."
)

WEAK_ANSWER = (
    "Um, I think it went pretty well overall, you know. We basically just sort of "
    "did the usual stuff, I guess. It was kind of a team effort really."
)


def test_vector_is_fixed_length_for_any_input() -> None:
    for text in ("", "   ", "one", WEAK_ANSWER, STRONG_ANSWER):
        assert len(features_to_vector(extract_features(text))) == N_FEATURES


def test_empty_answer_yields_all_zeros() -> None:
    features = extract_features("")
    assert set(features) == set(FEATURE_NAMES)
    assert all(value == 0.0 for value in features.values())


def test_no_feature_is_nan_or_infinite() -> None:
    vector = features_to_vector(extract_features(STRONG_ANSWER))
    assert np.isfinite(vector).all()


def test_strong_answer_beats_weak_on_structure_and_specificity() -> None:
    strong = extract_features(STRONG_ANSWER)
    weak = extract_features(WEAK_ANSWER)

    assert strong["star_coverage"] > weak["star_coverage"]
    assert strong["number_density"] > weak["number_density"]
    assert strong["named_entity_density"] > weak["named_entity_density"]


def test_weak_answer_has_higher_hedging_and_filler() -> None:
    strong = extract_features(STRONG_ANSWER)
    weak = extract_features(WEAK_ANSWER)

    assert weak["hedge_rate"] > strong["hedge_rate"]
    assert weak["filler_rate"] > strong["filler_rate"]


def test_star_coverage_requires_all_three_elements() -> None:
    """The geometric mean means one strong element cannot carry the score."""
    situation_only = extract_features(
        "When I was at my last company we were running a payments service."
    )
    all_three = extract_features(STRONG_ANSWER)

    assert situation_only["star_coverage"] < 0.3
    assert all_three["star_coverage"] > situation_only["star_coverage"]


def test_ownership_ratio_distinguishes_i_from_we() -> None:
    mine = extract_features("I built the service and I owned the rollout myself.")
    ours = extract_features("We built the service and our team owned the rollout.")

    assert mine["ownership_ratio"] > 0.9
    assert ours["ownership_ratio"] < 0.1


def test_typed_answer_reports_no_audio() -> None:
    features = extract_features(STRONG_ANSWER)
    assert features["has_audio"] == 0.0
    assert features["words_per_minute"] == 0.0


def test_word_timings_produce_prosody_features() -> None:
    words = STRONG_ANSWER.split(" ")
    timings = []
    cursor = 0.0
    for word in words:
        timings.append({"word": word, "start": cursor, "end": cursor + 0.4})
        cursor += 0.45

    features = extract_features(STRONG_ANSWER, word_timings=timings, audio_duration_s=cursor)

    assert features["has_audio"] == 1.0
    assert 100 < features["words_per_minute"] < 180
    assert features["silence_ratio"] >= 0.0


def test_long_pauses_are_detected() -> None:
    timings = [
        {"word": "I", "start": 0.0, "end": 0.3},
        {"word": "built", "start": 2.5, "end": 2.9},  # 2.2s gap
        {"word": "it", "start": 3.0, "end": 3.2},
    ]
    features = extract_features("I built it", word_timings=timings, audio_duration_s=3.2)

    assert features["long_pause_rate"] > 0
    assert features["mean_pause_s"] > 0.7


def test_malformed_timings_are_ignored_not_fatal() -> None:
    features = extract_features(
        "I built it",
        word_timings=[{"word": "I"}, {"start": "bad", "end": 1.0}, None],  # type: ignore[list-item]
        audio_duration_s=1.0,
    )
    assert features["has_audio"] == 0.0


def test_batch_extraction_matches_single_extraction() -> None:
    frame = pd.DataFrame({"answer": [STRONG_ANSWER, WEAK_ANSWER, ""]})
    batch = extract_batch(frame)

    assert list(batch.columns) == list(FEATURE_NAMES)
    assert len(batch) == 3

    single = extract_features(STRONG_ANSWER)
    for name in FEATURE_NAMES:
        assert batch.iloc[0][name] == pytest.approx(single[name])


def test_batch_requires_answer_column() -> None:
    with pytest.raises(ValueError, match="answer"):
        extract_batch(pd.DataFrame({"text": ["hello"]}))


# ─── Filler breakdown (display only) ──────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "It was, like, the hardest sprint we had.",
        "Like, we had no tests at all.",
        "We shipped it and like, nobody noticed.",
        "Um like we just rewrote it.",
        "It was like um two weeks late.",
        "I was like, like, really nervous.",
        "The service was, like you know, falling over.",
    ],
)
def test_filler_like_is_counted(text: str) -> None:
    assert filler_breakdown(text).get("like", 0) >= 1


@pytest.mark.parametrize(
    "text",
    [
        "I would like to lead a team next year.",
        "I like Python more than Java.",
        "We used tools like Redis and Kafka.",
        "It looks like the cache was the bottleneck.",
        "It took something like 20 percent longer.",
        "I feel like the design was right.",
        "Like I said, the rollout was staged.",
    ],
)
def test_non_filler_like_is_not_counted(text: str) -> None:
    assert "like" not in filler_breakdown(text)


def test_breakdown_counts_each_filler_and_sorts_by_count() -> None:
    text = "Um, so, like, we basically, um, had to, like, rewrite it, you know."
    assert filler_breakdown(text) == {"like": 2, "um": 2, "basically": 1, "you know": 1}


def test_breakdown_of_a_clean_or_empty_answer_is_empty() -> None:
    assert filler_breakdown("") == {}
    assert filler_breakdown(STRONG_ANSWER) == {}


def test_breakdown_never_changes_model_features() -> None:
    """The model's filler_rate keeps the lexicon it was trained on."""
    text = "It was, like, a hard project and, like, we were late."
    assert filler_breakdown(text)["like"] == 2
    assert extract_features(text)["filler_rate"] == 0.0
