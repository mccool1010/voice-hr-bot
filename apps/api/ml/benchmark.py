"""Benchmark the neural scorer against simpler baselines.

A neural network is only worth shipping if it beats what you would get for free.
This script trains ridge regression and gradient boosting on the identical
feature matrix and split, then prints all three side by side.

Run it whenever the feature set changes. If the MLP is not clearly ahead of
gradient boosting, the honest move is to ship the simpler model — and if all
three land in the same place, the limit is the features, not the architecture.

    python -m ml.benchmark
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from app.scoring.features import FEATURE_NAMES, extract_batch
from app.scoring.model import TARGET_NAMES
from ml.train_scorer import load_frame, split_frame, train

warnings.filterwarnings("ignore", category=UserWarning)


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("ml/data/synthetic.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()

    try:
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge
    except ImportError:
        raise SystemExit(
            "scikit-learn is required for the benchmark: pip install scikit-learn"
        ) from None

    frame = load_frame(args.data)
    train_df, val_df, test_df = split_frame(frame, args.seed)

    # The MLP trains on train+val (val is used for early stopping); the
    # baselines get train+val too, so the comparison is like for like.
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    x_fit = extract_batch(fit_df)[list(FEATURE_NAMES)].to_numpy(np.float32)
    x_test = extract_batch(test_df)[list(FEATURE_NAMES)].to_numpy(np.float32)

    print(f"Rows: fit={len(fit_df):,}  test={len(test_df):,}  features={x_fit.shape[1]}\n")

    results: dict[str, dict[str, float]] = {}

    for name, factory in (
        ("mean baseline", lambda: DummyRegressor(strategy="mean")),
        ("ridge", lambda: Ridge(alpha=1.0)),
        (
            "gradient boosting",
            lambda: GradientBoostingRegressor(
                n_estimators=300, max_depth=3, random_state=args.seed
            ),
        ),
    ):
        scores: dict[str, float] = {}
        for target in TARGET_NAMES:
            model = factory()
            model.fit(x_fit, fit_df[target].to_numpy(np.float32))
            scores[target] = _r2(test_df[target].to_numpy(np.float32), model.predict(x_test))
        results[name] = scores

    print("Training the neural scorer for comparison...")
    _, metrics = train(
        frame,
        epochs=args.epochs,
        batch_size=128,
        lr=2e-3,
        seed=args.seed,
        patience=25,
        device="cpu",
    )
    results["neural (MLP)"] = {t: metrics[f"r2_{t}"] for t in TARGET_NAMES}

    header = f"\n{'model':<20}" + "".join(f"{t:>14}" for t in TARGET_NAMES) + f"{'mean':>10}"
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        row = f"{name:<20}" + "".join(f"{scores[t]:>14.3f}" for t in TARGET_NAMES)
        print(row + f"{np.mean(list(scores.values())):>10.3f}")

    best_baseline = max(np.mean(list(results[n].values())) for n in results if n != "neural (MLP)")
    neural = float(np.mean(list(results["neural (MLP)"].values())))
    delta = neural - best_baseline

    print(f"\nNeural vs best baseline: {delta:+.3f} mean R²")
    if delta < 0.01:
        print(
            "The network is not beating the baselines. That is a feature-set limit,\n"
            "not an architecture problem — add signal before adding capacity."
        )


if __name__ == "__main__":
    main()
