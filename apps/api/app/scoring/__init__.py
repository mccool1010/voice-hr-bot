"""Answer scoring: feature engineering, PyTorch model, semantic relevance."""

from app.scoring.features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_batch,
    extract_features,
    features_to_vector,
)
from app.scoring.model import SCORER_VERSION, TARGET_NAMES, AnswerScorer
from app.scoring.service import ScoreResult, is_trained, score_answer, version

__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "SCORER_VERSION",
    "TARGET_NAMES",
    "AnswerScorer",
    "ScoreResult",
    "extract_batch",
    "extract_features",
    "features_to_vector",
    "is_trained",
    "score_answer",
    "version",
]
