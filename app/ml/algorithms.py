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
    # In Experiment Config form (lists or `{type, low, high, log}` ranges), so it can prefill
    # the UI and YAML as-is. Capped so a default random search (10 Candidates) takes ~30-60 s.
    # Empty means the algorithm only runs with fixed hyperparameters.
    search_space: dict[str, Any] = field(default_factory=dict)

    def param_names(self) -> set[str]:
        return set(self.make_estimator().get_params(deep=False))


ALGORITHMS: dict[str, Algorithm] = {
    "linear_regression": Algorithm(LinearRegression, scale_features=True),
    "random_forest": Algorithm(
        RandomForestRegressor,
        scale_features=False,
        default_params={"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2},
        search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 6, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 2, "high": 8},
            "max_features": [1.0, 0.5, "sqrt"],
        },
    ),
    "gradient_boosting": Algorithm(
        GradientBoostingRegressor,
        scale_features=False,
        default_params={"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
        search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "learning_rate": {"type": "float", "low": 0.03, "high": 0.3, "log": True},
            "max_depth": [2, 3, 4],
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        },
    ),
}
