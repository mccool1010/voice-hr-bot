"""NumPy inference for the answer scorer — no PyTorch at runtime.

The network is trained in PyTorch (`app/scoring/model.py`, `ml/train_scorer.py`)
and exported to a small `.npz` of weights. Serving it needs nothing but NumPy,
which keeps the deploy image under 512 MB of RAM instead of carrying a ~2 GB
deep-learning runtime to run a 24→64→32→5 MLP.

The forward pass mirrors `AnswerScorer.forward` exactly — same normalisation,
exact (erf-based) GELU, same sigmoid scaling — and a test asserts the two agree
to within float32 rounding.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.scoring.features import FEATURE_NAMES

_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching torch.nn.GELU() with its default approximate='none'."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class NumpyScorer:
    mean: np.ndarray
    std: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    heads_w: np.ndarray  # (n_targets, hidden)
    heads_b: np.ndarray  # (n_targets,)
    version: str = "unknown"
    feature_names: tuple[str, ...] = FEATURE_NAMES
    target_names: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(batch, n_features) → (batch, n_targets), each in 0-100."""
        h = (x.astype(np.float64) - self.mean) / self.std
        h = gelu(h @ self.w1.T + self.b1)
        h = gelu(h @ self.w2.T + self.b2)
        return sigmoid(h @ self.heads_w.T + self.heads_b) * 100.0

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """Accepts (n_features,) or (batch, n_features)."""
        x = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        return self.forward(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "target_names": list(self.target_names),
            "metrics": self.metrics,
        }
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            heads_w=self.heads_w,
            heads_b=self.heads_b,
            meta=np.array(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path: Path) -> NumpyScorer:
        with np.load(path, allow_pickle=False) as data:
            meta: dict[str, Any] = json.loads(str(data["meta"]))
            scorer = cls(
                mean=data["mean"],
                std=data["std"],
                w1=data["w1"],
                b1=data["b1"],
                w2=data["w2"],
                b2=data["b2"],
                heads_w=data["heads_w"],
                heads_b=data["heads_b"],
                version=meta.get("version", "unknown"),
                feature_names=tuple(meta.get("feature_names", FEATURE_NAMES)),
                target_names=tuple(meta.get("target_names", ())),
                metrics=meta.get("metrics", {}),
            )
        if scorer.feature_names != FEATURE_NAMES:
            raise ValueError(
                "Exported scorer feature layout does not match the current extractor. "
                "Retrain with: python -m ml.train_scorer"
            )
        return scorer
