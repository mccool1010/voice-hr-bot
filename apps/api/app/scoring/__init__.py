"""Answer scoring: feature engineering, PyTorch model, semantic relevance."""

from app.scoring.features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_batch,
    extract_features,
    features_to_vector,
)
from app.scoring.service import ScoreResult, is_trained, score_answer, version
from app.scoring.spec import SCORER_VERSION, TARGET_NAMES

__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "SCORER_VERSION",
    "TARGET_NAMES",
    "ScoreResult",
    "extract_batch",
    "extract_features",
    "features_to_vector",
    "is_trained",
    "score_answer",
    "version",
]
