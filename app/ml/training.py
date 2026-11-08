"""Evaluate Candidates with the fixed CV folds, then refit the winner and score the Test Set."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

from app.jobs.config import ExperimentConfig
from app.ml.data import Split, cv_folds
from app.ml.pipeline import build_pipeline

# Predictions come out in original units for every pipeline, so these scores are too.
_SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
_NEGATED = {"rmse", "mae"}


def _sign(metric: str) -> float:
    """scikit-learn negates error scores so higher is better; undo it to report them as errors."""
    return -1.0 if metric in _NEGATED else 1.0


@dataclass(frozen=True)
class CandidateResult:
    params: dict[str, Any]
    fold_scores: dict[str, np.ndarray]

    @property
    def cv_rmse_mean(self) -> float:
        return float(self.fold_scores["rmse"].mean())

    def cv_metrics(self) -> dict[str, float]:
        """`cv_<metric>_mean` and `cv_<metric>_std` across the folds."""
        metrics = {}
        for name, scores in self.fold_scores.items():
            metrics[f"cv_{name}_mean"] = float(scores.mean())
            metrics[f"cv_{name}_std"] = float(scores.std())
        return metrics


def evaluate_candidate(
    config: ExperimentConfig, params: dict[str, Any], split: Split
) -> CandidateResult:
    """Cross-validate one Candidate on the training portion only; the Test Set is not touched."""
    scores = cross_validate(
        build_pipeline(config, params),
        split.X_train,
        split.y_train,
        cv=cv_folds(),
        scoring=_SCORING,
        n_jobs=-1,
        error_score="raise",
    )
    fold_scores = {name: _sign(name) * scores[f"test_{name}"] for name in _SCORING}
    return CandidateResult(params=params, fold_scores=fold_scores)


def best_candidate(results: list[CandidateResult]) -> CandidateResult:
    return min(results, key=lambda result: result.cv_rmse_mean)


def refit_and_score(
    config: ExperimentConfig, params: dict[str, Any], split: Split
) -> tuple[RegressorMixin, dict[str, float]]:
    """Fit on the whole training portion, then score the Test Set once."""
    pipeline = build_pipeline(config, params).fit(split.X_train, split.y_train)
    return pipeline, {
        f"test_{name}": _sign(name) * get_scorer(scorer)(pipeline, split.X_test, split.y_test)
        for name, scorer in _SCORING.items()
    }
