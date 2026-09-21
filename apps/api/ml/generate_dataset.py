"""Synthetic training corpus for the answer scorer.

Why synthetic: a cold-start scorer needs labelled interview answers, and there
is no public corpus of interview responses graded on this rubric. Rather than
hand-label a few hundred rows, this script defines an explicit generative
process and samples from it.

The method matters for whether the resulting model means anything:

  1. Sample a latent quality vector q = (structure, specificity, clarity, depth)
     from a correlated multivariate distribution — real answers that are well
     structured also tend to be specific, so independent sampling would create
     a corpus the model could exploit in ways real data would not support.
  2. *Render* q into natural-language text: q drives which STAR phrases appear,
     how many concrete numbers and tool names are used, sentence length, and the
     rate of hedges and fillers.
  3. Optionally render word timings, with pace and pause structure correlated to
     clarity, simulating a spoken answer.
  4. Re-extract features from the rendered text with the production extractor.

Training then learns features → q. Because step 2 is lossy, saturating and
noisy, and step 4 never sees q, this is a real inverse problem rather than an
identity mapping. Measured held-out R² (8,000 rows, seed 42):

    model               mean R²   overall R²
    ridge                 0.524        0.605
    gradient boosting     0.536        0.615
    neural (MLP)          0.548        0.634

All three land close together, which says the ceiling is the information the
features retain, not model capacity — a wider network overfits without gaining
R². `python -m ml.benchmark` reproduces this table.

The model this produces is a *bootstrap*. `ml/export_dataset.py` replaces it
with real answers graded by the LLM rubric once interviews accumulate.

Usage:
    python -m ml.generate_dataset --rows 8000 --out ml/data/synthetic.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20240919

# ─── Text building blocks ─────────────────────────────────────────────────────

SITUATIONS = [
    "When I was at my last company we were running a payments service that kept timing out",
    "At my previous role the team had a data pipeline that broke roughly twice a week",
    "The project I owned was a customer dashboard that had grown unmaintainable",
    "My team inherited a legacy checkout flow with no tests at all",
    "We were migrating a monolith to services and the auth layer was the blocker",
    "The situation was that our onboarding funnel was dropping users at the third step",
]

ACTIONS = [
    "I designed a queue-backed retry layer and I implemented it over two sprints",
    "I built a validation stage in front of the pipeline and I wrote contract tests per source",
    "I proposed splitting the dashboard by domain and I led the refactor myself",
    "I decided to add characterisation tests first, then I refactored behind them",
    "I owned the auth extraction and I chose to run both paths in parallel during rollout",
    "I started by instrumenting each funnel step, then I rebuilt the worst one",
]

RESULTS = [
    "As a result timeouts dropped by 94 percent and we stopped getting paged at night",
    "The impact was that pipeline breakages went from twice a week to roughly once a quarter",
    "We shipped it in six weeks and page load improved from 4.2 seconds to 1.1 seconds",
    "The outcome was that we cut our regression escape rate by about 60 percent",
    "It ended up reducing auth latency by 180 milliseconds across every service",
    "Completion rate grew by 23 percent, which was worth around 400 extra signups a month",
]

DEPTH_CLAUSES = [
    "The trade-off was that retries added tail latency, so I capped the backoff",
    "I chose that over a cache because the read pattern was too sparse to hit",
    "What made it hard was that the failure was non-deterministic under load",
    "I was wrong about the root cause at first, and the profiler is what corrected me",
    "The reason that worked is that the contention was on write locks, not on CPU",
    "In hindsight I would have measured before refactoring rather than after",
]

TOOLS = [
    "PostgreSQL",
    "Redis",
    "Kafka",
    "Terraform",
    "Kubernetes",
    "React",
    "FastAPI",
    "Datadog",
    "Snowflake",
    "PyTorch",
    "Airflow",
    "Grafana",
]

VAGUE_FILLER = [
    "I think it went pretty well overall",
    "we basically just tried to do our best with it",
    "it was kind of a team effort really",
    "you know, it was sort of a normal project",
    "I mean, I guess it was fine",
    "we did the usual stuff you would expect",
]

HEDGE_INSERTS = ["I think", "I guess", "sort of", "kind of", "maybe", "probably"]
FILLER_INSERTS = ["um", "uh", "you know", "like", "basically", "actually"]


def _sample_quality(n: int, rng: np.random.Generator) -> np.ndarray:
    """Correlated latent quality in [0, 1], shape (n, 4).

    The covariance encodes that structure/specificity/depth move together more
    strongly than any of them moves with clarity — a candidate can be crisp but
    shallow, or deep but rambling.
    """
    mean = np.zeros(4)
    cov = np.array(
        [
            [1.00, 0.62, 0.35, 0.55],
            [0.62, 1.00, 0.30, 0.66],
            [0.35, 0.30, 1.00, 0.28],
            [0.55, 0.66, 0.28, 1.00],
        ]
    )
    latent = rng.multivariate_normal(mean, cov, size=n)
    # Logistic squash to [0, 1], then widen slightly so the tails are populated.
    quality = 1.0 / (1.0 + np.exp(-latent * 1.15))
    return np.clip(quality, 0.02, 0.98)


def _render_answer(q: np.ndarray, rng: np.random.Generator) -> str:
    """Turn a quality vector into plausible interview speech."""
    structure, specificity, clarity, depth = q
    parts: list[str] = []

    # Structure drives whether the STAR elements are present at all.
    if rng.random() < structure:
        parts.append(rng.choice(SITUATIONS))
    if rng.random() < structure * 0.95 + 0.05:
        parts.append(rng.choice(ACTIONS))
    if rng.random() < structure * 0.9:
        parts.append(rng.choice(RESULTS))

    # Depth adds reasoning clauses about trade-offs and mistakes.
    for _ in range(rng.binomial(3, depth)):
        parts.append(rng.choice(DEPTH_CLAUSES))

    # Specificity adds named tools and figures.
    n_tools = rng.binomial(3, specificity)
    if n_tools:
        chosen = rng.choice(TOOLS, size=min(n_tools, len(TOOLS)), replace=False)
        parts.append(f"We were using {', '.join(chosen)} for that")
    if rng.random() < specificity * 0.8:
        parts.append(
            f"It handled about {rng.integers(2, 900)}k requests a day "
            f"at around {rng.integers(20, 400)} milliseconds p99"
        )

    # Low quality fills the gap with content-free padding.
    for _ in range(rng.binomial(3, 1.0 - (structure + depth) / 2.0)):
        parts.append(rng.choice(VAGUE_FILLER))

    if not parts:
        parts.append(rng.choice(VAGUE_FILLER))

    # Poorly structured answers sometimes arrive out of order — result before
    # context, action before situation.
    if rng.random() < 0.12 * (1.0 - structure):
        rng.shuffle(parts)

    text = ". ".join(str(p) for p in parts) + "."

    # Low clarity injects hedges and fillers mid-sentence.
    noise_rate = (1.0 - clarity) * 0.16
    words = text.split(" ")
    out: list[str] = []
    for word in words:
        if rng.random() < noise_rate:
            pool = HEDGE_INSERTS if rng.random() < 0.5 else FILLER_INSERTS
            out.append(str(rng.choice(pool)))
        out.append(word)
    return " ".join(out)


def _render_timings(
    text: str, q: np.ndarray, rng: np.random.Generator
) -> tuple[list[dict[str, float]], float]:
    """Simulate Whisper word offsets for a spoken answer.

    Clarity drives pace stability: an unclear speaker pauses more often and more
    variably. Pace itself is sampled around a realistic 130 wpm.
    """
    _, _, clarity, _ = q
    words = text.split(" ")

    base_wpm = float(rng.normal(135, 22))
    base_wpm = float(np.clip(base_wpm, 75, 205))
    sec_per_word = 60.0 / base_wpm

    pause_prob = 0.04 + (1.0 - clarity) * 0.13
    pause_scale = 0.35 + (1.0 - clarity) * 0.9

    timings: list[dict[str, float]] = []
    cursor = round(float(rng.uniform(0.0, 0.4)), 3)

    for word in words:
        duration = max(0.06, float(rng.normal(sec_per_word, sec_per_word * 0.3)))
        start, end = cursor, cursor + duration
        timings.append(
            {"word": word, "start": round(start, 3), "end": round(end, 3), "probability": 0.95}
        )
        cursor = end
        if rng.random() < pause_prob:
            cursor += float(rng.exponential(pause_scale))

    return timings, round(cursor + 0.2, 3)


def build(rows: int, seed: int = RNG_SEED, audio_fraction: float = 0.65) -> pd.DataFrame:
    """Generate the corpus as a DataFrame."""
    rng = np.random.default_rng(seed)
    quality = _sample_quality(rows, rng)

    records: list[dict[str, object]] = []
    for i in range(rows):
        q = quality[i]
        answer = _render_answer(q, rng)

        word_timings: list[dict[str, float]] | None = None
        duration: float | None = None
        if rng.random() < audio_fraction:
            word_timings, duration = _render_timings(answer, q, rng)

        structure, specificity, clarity, depth = (q * 100.0).tolist()
        # Overall is a weighted blend plus label noise, mirroring the fact that
        # two human graders never agree exactly.
        overall = float(
            np.clip(
                0.28 * structure
                + 0.27 * specificity
                + 0.18 * clarity
                + 0.27 * depth
                + rng.normal(0, 4.5),
                0.0,
                100.0,
            )
        )

        records.append(
            {
                "answer": answer,
                "word_timings": word_timings,
                "audio_duration_s": duration,
                "structure": round(structure, 3),
                "specificity": round(specificity, 3),
                "clarity": round(clarity, 3),
                "depth": round(depth, 3),
                "overall": round(overall, 3),
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--out", type=Path, default=Path("ml/data/synthetic.parquet"))
    args = parser.parse_args()

    frame = build(args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.suffix == ".parquet":
        try:
            frame.to_parquet(args.out, index=False)
        except ImportError:
            args.out = args.out.with_suffix(".jsonl")
            frame.to_json(args.out, orient="records", lines=True)
    else:
        frame.to_json(args.out, orient="records", lines=True)

    print(f"Wrote {len(frame):,} rows to {args.out}")
    print(frame[["structure", "specificity", "clarity", "depth", "overall"]].describe().round(1))


if __name__ == "__main__":
    main()
