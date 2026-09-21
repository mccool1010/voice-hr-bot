"""Can embedding similarity tell an on-topic answer from an off-topic one?

An earlier version of the scorer multiplied every answer's score by a factor
derived from question-answer cosine similarity, on the assumption that a low
similarity meant the candidate had drifted off the question. In the first real
spoken test, a perfectly reasonable answer scored 0.0 similarity and had its
score cut by two thirds. This script measures whether the assumption holds.

It embeds three groups of question-answer pairs and prints each group's
similarity distribution:

  on-topic                 a genuine answer to the question
  other interview answer   a genuine answer written for a *different* question
  unrelated / empty        small talk, or "I don't know"

    python -m ml.calibrate_relevance
    python -m ml.calibrate_relevance --model sentence-transformers/all-MiniLM-L6-v2

Findings (all three models tested overlap too much to act as a gate):

    model                     on-topic min   other-answer max   unrelated max
    all-MiniLM-L6-v2              -0.008            0.431              0.208
    multi-qa-MiniLM-L6-cos-v1      0.024            0.391              0.228
    multi-qa-mpnet-base-cos-v1     0.023            0.396              0.209

Behavioural questions are generic ("tell me about a project") while answers are
specific stories, and one good story often answers several questions. So the
score no longer uses relevance at all; off-topic answers are penalised by the
LLM rubric, whose anchors score them 0-39. Relevance is still recorded and
shown to the candidate as information.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

QA_PAIRS: list[tuple[str, str]] = [
    (
        "To start, walk me through your background and what drew you to this role.",
        "At my last company our payments service kept timing out. I designed a retry layer "
        "in Redis and cut timeouts by 94 percent. That kind of reliability work is why this "
        "role appeals to me.",
    ),
    (
        "To start, walk me through your background and what drew you to this role.",
        "I studied computer science, then spent three years as a backend engineer at a "
        "fintech, mostly on Python services. I want to work somewhere with bigger scale.",
    ),
    (
        "Tell me about a recent project you're proud of. What was your part in it?",
        "I led the migration of our monolith's auth layer to its own service. I wrote the "
        "rollout plan and ran both paths in parallel for a month.",
    ),
    (
        "Tell me about a recent project you're proud of. What was your part in it?",
        "We rebuilt the customer dashboard in React. I owned the charts and cut page load "
        "from four seconds to one.",
    ),
    (
        "Describe the hardest technical problem you've solved. How did you approach it?",
        "A race condition corrupted ledger entries under load. I reproduced it with a stress "
        "test, found the missing lock, and added an idempotency key.",
    ),
    (
        "Describe the hardest technical problem you've solved. How did you approach it?",
        "Our ML model drifted badly after a data source changed. I built monitoring on "
        "feature distributions and retrained on the new data.",
    ),
    (
        "Tell me about a time something you owned went wrong. What did you do?",
        "I shipped a migration that locked a table in production for ten minutes. I rolled "
        "it back, wrote the postmortem, and added a check to CI.",
    ),
    (
        "Tell me about a time something you owned went wrong. What did you do?",
        "A release I owned broke checkout for mobile users. I paged the team, reverted within "
        "twenty minutes and we added device tests.",
    ),
    (
        "Describe a disagreement with a teammate and how it was resolved.",
        "A colleague wanted to rewrite the service in Go. I disagreed, so we benchmarked both "
        "and agreed to optimise the Python hot path instead.",
    ),
    (
        "Describe a disagreement with a teammate and how it was resolved.",
        "My designer and I disagreed about a settings page. We ran a quick user test with "
        "five people and went with what they preferred.",
    ),
    (
        "What are you trying to get better at right now, and how?",
        "System design. I'm reading Designing Data-Intensive Applications and writing a "
        "design doc for each project before I build it.",
    ),
    (
        "What are you trying to get better at right now, and how?",
        "Public speaking. I've started presenting at our internal tech talks once a month.",
    ),
]

UNRELATED = [
    "My favourite recipe is lasagne with extra basil and a lot of parmesan.",
    "The weather has been rainy all week so I have mostly stayed indoors.",
    "I support Manchester United and watch every match with my brother.",
    "I don't know.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    def similarity(question: str, answer: str) -> float:
        q, a = model.encode([question, answer], normalize_embeddings=True)
        return float(np.dot(q, a))

    questions = sorted({q for q, _ in QA_PAIRS})
    answers = [a for _, a in QA_PAIRS]
    genuine = set(QA_PAIRS)

    groups = {
        "on-topic": [similarity(q, a) for q, a in QA_PAIRS],
        "other interview answer": [
            similarity(q, a)
            for q, a in itertools.product(questions, answers)
            if (q, a) not in genuine
        ],
        "unrelated / empty": [similarity(q, a) for q in questions for a in UNRELATED],
    }

    print(f"model: {args.model}\n")
    for name, values in groups.items():
        xs = np.asarray(values)
        print(
            f"{name:24s} n={len(xs):3d}  min {xs.min():6.3f}  p10 {np.percentile(xs, 10):6.3f}"
            f"  median {np.median(xs):6.3f}  max {xs.max():6.3f}"
        )

    overlap = max(groups["other interview answer"]) > min(groups["on-topic"])
    print(
        "\nOn-topic and off-topic ranges overlap — similarity cannot gate the score."
        if overlap
        else "\nRanges separate — a threshold between them could gate the score."
    )


if __name__ == "__main__":
    main()
