"""The fixed list of algorithms a Training Job can use, with their default hyperparameters."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class Algorithm:
    make_estimator: Callable[[], RegressorMixin]
    # Only scale-sensitive estimators get a StandardScaler; trees don't need one.
    scale_features: bool
    # Defaults are capped so a fixed job finishes in seconds on a laptop.
    default_params: dict[str, Any] = field(default_factory=dict)

    def param_names(self) -> set[str]:
        return set(self.make_estimator().get_params(deep=False))


ALGORITHMS: dict[str, Algorithm] = {
    "linear_regression": Algorithm(LinearRegression, scale_features=True),
    "random_forest": Algorithm(
        RandomForestRegressor,
        scale_features=False,
        default_params={"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2},
    ),
    "gradient_boosting": Algorithm(
        GradientBoostingRegressor,
        scale_features=False,
        default_params={"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    ),
}
