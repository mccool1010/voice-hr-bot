"""The NumPy export must predict exactly what the PyTorch model predicts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.scoring.features import FEATURE_NAMES, N_FEATURES
from app.scoring.numpy_model import NumpyScorer, gelu

torch = pytest.importorskip("torch")

from app.scoring.model import AnswerScorer, ScorerCheckpoint, export_numpy, predict


def _trained_like_model() -> AnswerScorer:
    """Random weights and non-trivial normalisation — parity must hold for any."""
    torch.manual_seed(7)
    model = AnswerScorer()
    rng = np.random.default_rng(7)
    model.set_normalisation(rng.normal(5, 3, N_FEATURES), rng.uniform(0.5, 4, N_FEATURES))
    return model.eval()


def test_gelu_matches_torch() -> None:
    x = np.linspace(-6, 6, 101)
    expected = torch.nn.functional.gelu(torch.tensor(x)).numpy()
    np.testing.assert_allclose(gelu(x), expected, atol=1e-7)


def test_numpy_export_matches_torch_predictions(tmp_path: Path) -> None:
    model = _trained_like_model()
    npz = export_numpy(
        model, ScorerCheckpoint(state_dict=model.state_dict(), version="test"), tmp_path / "s.npz"
    )
    scorer = NumpyScorer.load(npz)

    x = np.random.default_rng(1).normal(0, 20, (64, N_FEATURES)).astype(np.float32)
    np.testing.assert_allclose(scorer.predict(x), predict(model, x), atol=1e-3)
    assert scorer.version == "test"


def test_export_rejects_feature_drift(tmp_path: Path) -> None:
    model = _trained_like_model()
    checkpoint = ScorerCheckpoint(
        state_dict=model.state_dict(), feature_names=(*FEATURE_NAMES[:-1], "renamed")
    )
    npz = export_numpy(model, checkpoint, tmp_path / "old.npz")
    with pytest.raises(ValueError, match="feature layout"):
        NumpyScorer.load(npz)
