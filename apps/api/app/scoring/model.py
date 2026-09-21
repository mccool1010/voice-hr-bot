"""PyTorch answer-quality scorer.

A small multi-task MLP: one shared trunk over the engineered feature vector,
five regression heads (four rubric dimensions plus an overall score). Multi-task
rather than a single output because the dimensions are correlated but not
identical — sharing a trunk regularises each head, and the per-dimension
predictions are what the UI actually renders.

The model is deliberately small. With ~24 informative features and a few
thousand training rows, a 24→64→32 trunk is the right capacity; anything larger
memorises. Normalisation statistics are stored inside the checkpoint so
inference can never drift from training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from app.scoring.features import FEATURE_NAMES, N_FEATURES
from app.scoring.spec import N_TARGETS, SCORER_VERSION, TARGET_NAMES

__all__ = [
    "N_TARGETS",
    "SCORER_VERSION",
    "TARGET_NAMES",
    "AnswerScorer",
    "ScorerCheckpoint",
    "export_numpy",
    "load_scorer",
    "predict",
]


class AnswerScorer(nn.Module):
    """Shared trunk, one linear head per rubric dimension."""

    # Registered as buffers in __init__; declared here so type checkers know
    # they are tensors rather than arbitrary module attributes.
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(nn.Linear(h2, 1) for _ in range(N_TARGETS))

        # Registered as buffers so they move with `.to(device)` and are saved in
        # the state dict — inference cannot accidentally use different stats.
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))

    def set_normalisation(self, mean: np.ndarray, std: np.ndarray) -> None:
        # Guard against zero-variance features (e.g. `has_audio` in a text-only set).
        safe_std = np.where(std < 1e-6, 1.0, std)
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feature_std.copy_(torch.as_tensor(safe_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_features) → (batch, n_targets), each in 0-100."""
        normalised = (x - self.feature_mean) / self.feature_std
        shared = self.trunk(normalised)
        stacked = torch.cat([head(shared) for head in self.heads], dim=-1)
        return torch.sigmoid(stacked) * 100.0


@dataclass
class ScorerCheckpoint:
    """Everything needed to reproduce a prediction."""

    state_dict: dict[str, Any]
    version: str = SCORER_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES
    target_names: tuple[str, ...] = TARGET_NAMES
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict,
                "version": self.version,
                "feature_names": list(self.feature_names),
                "target_names": list(self.target_names),
                "metrics": self.metrics,
                "trained_at": self.trained_at,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> ScorerCheckpoint:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            state_dict=blob["state_dict"],
            version=blob.get("version", "unknown"),
            feature_names=tuple(blob.get("feature_names", FEATURE_NAMES)),
            target_names=tuple(blob.get("target_names", TARGET_NAMES)),
            metrics=blob.get("metrics", {}),
            trained_at=blob.get("trained_at"),
        )


def load_scorer(path: Path, device: str = "cpu") -> tuple[AnswerScorer, ScorerCheckpoint]:
    """Load a trained scorer, refusing a checkpoint whose features have drifted."""
    checkpoint = ScorerCheckpoint.load(path)

    if tuple(checkpoint.feature_names) != FEATURE_NAMES:
        raise ValueError(
            "Checkpoint feature layout does not match the current extractor.\n"
            f"  checkpoint: {len(checkpoint.feature_names)} features\n"
            f"  current:    {len(FEATURE_NAMES)} features\n"
            "Retrain with: python -m ml.train_scorer"
        )

    model = AnswerScorer(n_features=len(checkpoint.feature_names))
    model.load_state_dict(checkpoint.state_dict)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model: AnswerScorer, vectors: np.ndarray) -> np.ndarray:
    """Score a batch. Accepts (n_features,) or (batch, n_features)."""
    tensor = torch.as_tensor(vectors, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return model(tensor.to(next(model.parameters()).device)).cpu().numpy()


def export_numpy(model: AnswerScorer, checkpoint: ScorerCheckpoint, path: Path) -> Path:
    """Write the weights as an .npz that `NumpyScorer` can serve without torch."""
    from app.scoring.numpy_model import NumpyScorer

    state = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    heads = sorted({k.split(".")[1] for k in state if k.startswith("heads.")}, key=int)
    NumpyScorer(
        mean=state["feature_mean"],
        std=state["feature_std"],
        w1=state["trunk.0.weight"],
        b1=state["trunk.0.bias"],
        w2=state["trunk.3.weight"],
        b2=state["trunk.3.bias"],
        heads_w=np.concatenate([state[f"heads.{i}.weight"] for i in heads], axis=0),
        heads_b=np.concatenate([state[f"heads.{i}.bias"] for i in heads], axis=0),
        version=checkpoint.version,
        feature_names=tuple(checkpoint.feature_names),
        target_names=tuple(checkpoint.target_names),
        metrics=checkpoint.metrics,
    ).save(path)
    return path
