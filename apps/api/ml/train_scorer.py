"""Train the answer-quality scorer.

    python -m ml.generate_dataset --rows 8000
    python -m ml.train_scorer --epochs 120

Reports per-dimension R² and MAE on a held-out test split, and writes a
checkpoint containing the weights, the normalisation statistics, the feature
layout and the metrics — everything needed to reproduce a prediction.

Design notes:
  * Normalisation statistics are computed on the training split only and stored
    as model buffers, so inference can never use different statistics.
  * The loss is Huber rather than MSE: rubric labels have heavy tails and a few
    extreme answers should not dominate the gradient.
  * Early stopping is on validation loss with weight restoration, so `--epochs`
    is an upper bound rather than a target.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.scoring.features import FEATURE_NAMES, extract_batch
from app.scoring.model import (
    SCORER_VERSION,
    TARGET_NAMES,
    AnswerScorer,
    ScorerCheckpoint,
)

DEFAULT_DATA = Path("ml/data/synthetic.parquet")
DEFAULT_OUT = Path("ml/artifacts/answer_scorer.pt")


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        alt = path.with_suffix(".jsonl")
        if alt.exists():
            path = alt
        else:
            raise SystemExit(
                f"No dataset at {path}. Generate one first:\n"
                "  python -m ml.generate_dataset --rows 8000"
            )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, lines=True)


def split_frame(
    frame: pd.DataFrame, seed: int, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return (
        shuffled.iloc[n_test + n_val :].reset_index(drop=True),
        shuffled.iloc[n_test : n_test + n_val].reset_index(drop=True),
        shuffled.iloc[:n_test].reset_index(drop=True),
    )


def to_tensors(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    features = extract_batch(frame)
    x = torch.as_tensor(features[list(FEATURE_NAMES)].to_numpy(np.float32))
    y = torch.as_tensor(frame[list(TARGET_NAMES)].to_numpy(np.float32))
    return x, y


def evaluate(model: AnswerScorer, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Per-dimension R² and MAE on the given split."""
    model.eval()
    with torch.inference_mode():
        predictions = model(x.to(next(model.parameters()).device)).cpu().numpy()
    actual = y.numpy()

    metrics: dict[str, float] = {}
    for i, name in enumerate(TARGET_NAMES):
        pred_i, true_i = predictions[:, i], actual[:, i]
        residual = float(np.sum((true_i - pred_i) ** 2))
        total = float(np.sum((true_i - true_i.mean()) ** 2))
        metrics[f"r2_{name}"] = round(1.0 - residual / total if total > 0 else 0.0, 4)
        metrics[f"mae_{name}"] = round(float(np.mean(np.abs(true_i - pred_i))), 3)

    metrics["r2_mean"] = round(float(np.mean([metrics[f"r2_{n}"] for n in TARGET_NAMES])), 4)
    return metrics


def train(
    frame: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    patience: int,
    device: str,
    hidden: tuple[int, int] = (64, 32),
    dropout: float = 0.15,
) -> tuple[AnswerScorer, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df, val_df, test_df = split_frame(frame, seed)
    print(f"Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    print("Extracting features...")
    x_train, y_train = to_tensors(train_df)
    x_val, y_val = to_tensors(val_df)
    x_test, y_test = to_tensors(test_df)

    model = AnswerScorer(n_features=len(FEATURE_NAMES), hidden=hidden, dropout=dropout)
    # Statistics from the training split only — using the full set would leak.
    model.set_normalisation(mean=x_train.numpy().mean(axis=0), std=x_train.numpy().std(axis=0))
    model.to(device)

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    # delta=5.0 in score units: errors under 5 points are treated quadratically,
    # larger ones linearly, which stops outlier labels dominating.
    criterion = nn.HuberLoss(delta=5.0)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item() * xb.size(0)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            val_loss = criterion(model(x_val.to(device)), y_val.to(device)).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}  train {epoch_loss / len(train_df):7.3f}"
                f"  val {val_loss:7.3f}{'  *' if stale == 0 else ''}"
            )

        if stale >= patience:
            print(f"  early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate(model, x_test, y_test)
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden", type=int, nargs=2, default=(64, 32))
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    frame = load_frame(args.data)
    print(f"Loaded {len(frame):,} rows from {args.data}")

    model, metrics = train(
        frame,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        device=args.device,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
    )

    print("\nHeld-out test metrics:")
    for name in TARGET_NAMES:
        print(f"  {name:<12}  R2 {metrics[f'r2_{name}']:6.3f}   MAE {metrics[f'mae_{name}']:6.2f}")
    print(f"  {'mean R2':<12}  {metrics['r2_mean']:6.3f}")

    model.cpu()
    ScorerCheckpoint(
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        version=SCORER_VERSION,
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        metrics=metrics,
        trained_at=datetime.now(UTC).isoformat(),
    ).save(args.out)

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved checkpoint to {args.out}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
