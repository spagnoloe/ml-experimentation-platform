"""Build the full, unfitted pipeline for one Candidate of an Experiment Config.

Everything that learns from data (scaling, the estimator, the target transform) lives inside
this one object, so it is fitted only on the rows it is trained on and persisted as a whole.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms import ALGORITHMS

if TYPE_CHECKING:
    from app.jobs.config import ExperimentConfig


# Types a fitted pipeline contains that skops does not trust by default. Logged models are
# serialized with skops, which refuses to load any type not listed here (notebook issue 9).
SKOPS_TRUSTED_TYPES: tuple[str, ...] = (
    "app.ml.pipeline.BedroomsPerRoom",
    # Node storage of the tree ensembles; only files produced by our own Runs are loaded.
    "sklearn.tree._tree.Tree",
)


class BedroomsPerRoom(TransformerMixin, BaseEstimator):
    """Append `BedroomsPerRoom = AveBedrms / AveRooms`, 0 where a district has no rooms."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BedroomsPerRoom":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rooms = X["AveRooms"].to_numpy(dtype="float64")
        bedrooms = X["AveBedrms"].to_numpy(dtype="float64")
        ratio = np.divide(bedrooms, rooms, out=np.zeros_like(rooms), where=rooms != 0)
        return X.assign(BedroomsPerRoom=ratio)


def build_pipeline(config: "ExperimentConfig", params: dict[str, Any]) -> RegressorMixin:
    algorithm = ALGORITHMS[config.algorithm]
    estimator = algorithm.make_estimator()
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=config.random_state)
    estimator.set_params(**params)

    steps: list[tuple[str, Any]] = []
    if config.features.bedrooms_per_room:
        steps.append(("bedrooms_per_room", BedroomsPerRoom()))
    if algorithm.scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", estimator))
    pipeline = Pipeline(steps)

    if config.target_transform == "log1p":
        # Predictions come back through expm1, so they and every metric are in original units.
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        )
    return pipeline
