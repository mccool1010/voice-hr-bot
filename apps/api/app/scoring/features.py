"""Feature engineering for interview answers.

Turns raw text (plus optional Whisper word timings) into a fixed-length numeric
vector. The same code path runs at training time over a DataFrame of thousands
of answers and at inference time over a single answer, which is what keeps the
two from drifting apart.

Feature groups:
  lexical    — length, diversity, sentence structure
  structure  — STAR signals (situation / action / result) and ownership
  specificity— numbers, named entities, concrete detail
  hedging    — filler words, hedges, confidence markers
  prosody    — pace, pauses and hesitation, from Whisper word timings
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ─── Lexicons ─────────────────────────────────────────────────────────────────
# Kept as frozensets of lowercase tokens/phrases. Phrase entries are matched on
# the normalised string; single tokens on the token list.

SITUATION_MARKERS = (
    "when i",
    "at my",
    "we were",
    "the project",
    "my team",
    "i was working",
    "in my role",
    "last year",
    "at the time",
    "the situation",
    "we had to",
    "i joined",
    "the company",
    "our client",
    "the challenge was",
)
ACTION_MARKERS = (
    "i built",
    "i designed",
    "i implemented",
    "i led",
    "i decided",
    "i wrote",
    "i created",
    "i refactored",
    "i proposed",
    "i migrated",
    "i set up",
    "i introduced",
    "i automated",
    "i debugged",
    "i owned",
    "i chose",
    "my approach",
    "what i did",
    "i started by",
)
RESULT_MARKERS = (
    "resulted in",
    "reduced",
    "increased",
    "improved",
    "cut down",
    "saved",
    "the impact",
    "as a result",
    "we shipped",
    "went from",
    "ended up",
    "which meant",
    "the outcome",
    "grew by",
    "dropped by",
    "sped up",
)

HEDGES = frozenset(
    {
        "maybe",
        "probably",
        "possibly",
        "somewhat",
        "arguably",
        "presumably",
        "roughly",
        "approximately",
        "sorta",
        "kinda",
        "perhaps",
        "supposedly",
    }
)
HEDGE_PHRASES = (
    "i think",
    "i guess",
    "i suppose",
    "sort of",
    "kind of",
    "or something",
    "i'm not sure",
    "im not sure",
    "i would say",
    "more or less",
    "if i remember",
)

FILLERS = frozenset({"um", "uh", "er", "ah", "hmm", "mm", "eh", "uhh", "umm"})
FILLER_PHRASES = ("you know", "i mean", "like i said", "basically", "actually", "literally")

# Words after which "like" is a verb or a comparison, never a filler:
# "I would like", "I like", "looks like", "feel like", "something like".
_LIKE_NOT_FILLER_AFTER = frozenset(
    {
        "i", "we", "you", "they", "he", "she", "would", "i'd", "we'd", "you'd", "don't",
        "didn't", "doesn't", "not", "to", "really", "also", "just", "feel", "feels", "felt",
        "look", "looks", "looked", "seem", "seems", "seemed", "sound", "sounds", "something",
        "anything", "nothing", "things", "stuff", "more", "much", "exactly",
    }
)  # fmt: skip
_CLAUSE_OPENERS = frozenset({"and", "so", "but", "or"})
_TOKEN_RE = re.compile(r"[a-z']+|[.,!?;:]")

CONFIDENCE_MARKERS = (
    "i'm confident",
    "im confident",
    "definitely",
    "certainly",
    "without doubt",
    "i know",
    "clearly",
    "absolutely",
    "i made sure",
    "i verified",
)

_WORD_RE = re.compile(r"[a-zA-Z']+")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_PERCENT_RE = re.compile(r"\d+\s*(?:%|percent)")
# A capitalised word not at sentence start — a cheap proxy for named tools,
# companies and technologies without shipping an NER model.
_MID_CAPS_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z0-9+#.]{1,}\b", re.MULTILINE)

LONG_PAUSE_S = 0.7

FEATURE_NAMES: tuple[str, ...] = (
    # lexical
    "word_count",
    "sentence_count",
    "avg_sentence_len",
    "type_token_ratio",
    "mean_word_len",
    "long_word_ratio",
    # structure
    "situation_score",
    "action_score",
    "result_score",
    "star_coverage",
    "first_person_ratio",
    "ownership_ratio",
    # specificity
    "number_density",
    "percent_count",
    "named_entity_density",
    # hedging
    "hedge_rate",
    "filler_rate",
    "confidence_rate",
    # prosody
    "words_per_minute",
    "long_pause_rate",
    "mean_pause_s",
    "pause_variability",
    "silence_ratio",
    "has_audio",
)

N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float

    @classmethod
    def from_dicts(cls, raw: list[dict[str, Any]] | None) -> list[WordTiming]:
        if not raw:
            return []
        out: list[WordTiming] = []
        for item in raw:
            try:
                out.append(
                    cls(
                        word=str(item.get("word", "")).strip(),
                        start=float(item["start"]),
                        end=float(item["end"]),
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                # Whisper output is trusted, but timings can also arrive from a
                # client; one malformed entry must not lose the whole answer.
                continue
        return out


def _is_filler_like(tokens: list[str], i: int) -> bool:
    """Whether the "like" at ``tokens[i]`` is a filler rather than a verb or comparison.

    Only positions that are fillers in practice count, so "I like Python" and
    "tools like Redis" never do: set off by commas ("it was, like, hard"), opening
    a clause ("Like, we..." / "and like, we..."), beside another filler ("um like",
    "like, you know"), or repeated ("like like").
    """
    prev = tokens[i - 1] if i > 0 else None
    nxt = tokens[i + 1] if i + 1 < len(tokens) else None
    nxt2 = tokens[i + 2] if i + 2 < len(tokens) else None
    if prev in _LIKE_NOT_FILLER_AFTER or (nxt == "i" and nxt2 == "said"):
        return False  # "like I said" is counted once, as a phrase
    if prev in FILLERS or prev == "like" or nxt in FILLERS:
        return True
    if nxt == "you" and nxt2 == "know":
        return True
    return nxt == "," and (
        prev is None or prev in {".", "!", "?", ",", ";"} or prev in _CLAUSE_OPENERS
    )


def filler_breakdown(text: str) -> dict[str, int]:
    """Count each filler word or phrase in an answer, most frequent first.

    For showing the candidate what was heard, not for scoring. ``filler_rate``
    feeds the trained model, so its lexicon stays exactly as the checkpoint was
    trained; this breakdown uses the same lexicon and also counts "like" when it
    is used as a filler. It never changes a score.
    """
    lower = text.lower()
    tokens = _TOKEN_RE.findall(lower)
    counts: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        if tok in FILLERS or (tok == "like" and _is_filler_like(tokens, i)):
            counts[tok] = counts.get(tok, 0) + 1
    words_only = " ".join(t for t in tokens if t not in ".,!?;:")
    for phrase in FILLER_PHRASES:
        hits = len(re.findall(rf"\b{re.escape(phrase)}\b", words_only))
        if hits:
            counts[phrase] = counts.get(phrase, 0) + hits
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _count_phrases(text_lower: str, phrases: tuple[str, ...]) -> int:
    return sum(text_lower.count(phrase) for phrase in phrases)


def _saturating(count: int, scale: float) -> float:
    """Map an unbounded count onto 0-1 with diminishing returns.

    Three situation markers should not score three times as high as one; what
    matters is whether the element is present at all, with some credit for more.
    """
    return float(1.0 - np.exp(-count / scale))


def extract_features(
    answer: str,
    *,
    word_timings: list[dict[str, Any]] | None = None,
    audio_duration_s: float | None = None,
) -> dict[str, float]:
    """Compute the full feature dict for one answer.

    Always returns every key in `FEATURE_NAMES`, so the vector is fixed-length
    even for an empty answer.
    """
    text = (answer or "").strip()
    lower = text.lower()
    words = _WORD_RE.findall(lower)
    n_words = len(words)

    if n_words == 0:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    per_100 = 100.0 / n_words

    # ─── Lexical ──────────────────────────────────────────────────────────────
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    word_lengths = np.array([len(w) for w in words], dtype=np.float64)

    features: dict[str, float] = {
        "word_count": float(n_words),
        "sentence_count": float(n_sentences),
        "avg_sentence_len": float(n_words / n_sentences),
        "type_token_ratio": float(len(set(words)) / n_words),
        "mean_word_len": float(word_lengths.mean()),
        "long_word_ratio": float((word_lengths > 6).mean()),
    }

    # ─── Structure (STAR) ─────────────────────────────────────────────────────
    situation = _saturating(_count_phrases(lower, SITUATION_MARKERS), 1.5)
    action = _saturating(_count_phrases(lower, ACTION_MARKERS), 1.5)
    result = _saturating(_count_phrases(lower, RESULT_MARKERS), 1.5)

    i_count = words.count("i") + words.count("i'd") + words.count("i'll") + words.count("i've")
    we_count = words.count("we") + words.count("our") + words.count("us")

    features |= {
        "situation_score": situation,
        "action_score": action,
        "result_score": result,
        # Rewards answers that cover all three, not one element heavily.
        "star_coverage": float(np.cbrt(situation * action * result)),
        "first_person_ratio": float(i_count * per_100),
        # Near 1.0 means "I did"; near 0 means "the team did" — an ownership signal.
        "ownership_ratio": float(i_count / (i_count + we_count)) if (i_count + we_count) else 0.0,
    }

    # ─── Specificity ──────────────────────────────────────────────────────────
    features |= {
        "number_density": float(len(_NUMBER_RE.findall(text)) * per_100),
        "percent_count": float(len(_PERCENT_RE.findall(lower))),
        "named_entity_density": float(len(_MID_CAPS_RE.findall(text)) * per_100),
    }

    # ─── Hedging and confidence ───────────────────────────────────────────────
    hedge_hits = sum(1 for w in words if w in HEDGES) + _count_phrases(lower, HEDGE_PHRASES)
    filler_hits = sum(1 for w in words if w in FILLERS) + _count_phrases(lower, FILLER_PHRASES)

    features |= {
        "hedge_rate": float(hedge_hits * per_100),
        "filler_rate": float(filler_hits * per_100),
        "confidence_rate": float(_count_phrases(lower, CONFIDENCE_MARKERS) * per_100),
    }

    # ─── Prosody ──────────────────────────────────────────────────────────────
    features |= _prosody_features(WordTiming.from_dicts(word_timings), audio_duration_s, n_words)

    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def _prosody_features(
    timings: list[WordTiming], audio_duration_s: float | None, n_words: int
) -> dict[str, float]:
    """Delivery signals derived from Whisper word offsets.

    Returns zeros with `has_audio=0` for typed answers, so the model can learn to
    ignore this block rather than treating a typed answer as a silent one.
    """
    zeros = {
        "words_per_minute": 0.0,
        "long_pause_rate": 0.0,
        "mean_pause_s": 0.0,
        "pause_variability": 0.0,
        "silence_ratio": 0.0,
        "has_audio": 0.0,
    }
    if len(timings) < 2:
        return zeros

    starts = np.array([t.start for t in timings], dtype=np.float64)
    ends = np.array([t.end for t in timings], dtype=np.float64)

    # Gap between the end of each word and the start of the next.
    gaps = starts[1:] - ends[:-1]
    gaps = gaps[gaps >= 0]

    duration = audio_duration_s or float(ends[-1] - starts[0])
    if duration <= 0:
        return zeros

    speaking_time = float(np.sum(ends - starts))

    return {
        "words_per_minute": float(n_words / duration * 60.0),
        "long_pause_rate": float(np.sum(gaps > LONG_PAUSE_S) / duration * 60.0),
        "mean_pause_s": float(gaps.mean()) if gaps.size else 0.0,
        "pause_variability": float(gaps.std()) if gaps.size > 1 else 0.0,
        "silence_ratio": float(max(0.0, 1.0 - speaking_time / duration)),
        "has_audio": 1.0,
    }


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Ordered float32 vector. Order is `FEATURE_NAMES` and must never change."""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)


def extract_batch(answers: pd.DataFrame) -> pd.DataFrame:
    """Vectorised extraction over a DataFrame — used by the training pipeline.

    Expects columns: `answer`, and optionally `word_timings` and
    `audio_duration_s`. Returns one column per feature, index-aligned to input.
    """
    if "answer" not in answers.columns:
        raise ValueError("extract_batch requires an 'answer' column")

    has_timings = "word_timings" in answers.columns
    has_duration = "audio_duration_s" in answers.columns

    records = [
        extract_features(
            row.answer,
            word_timings=row.word_timings if has_timings else None,
            audio_duration_s=row.audio_duration_s if has_duration else None,
        )
        for row in answers.itertuples()
    ]
    return pd.DataFrame.from_records(records, index=answers.index, columns=list(FEATURE_NAMES))
