"""The Experiment Config: the data scientist's validated description of what to train."""

import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler

from app.ml.algorithms import ALGORITHMS

AlgorithmName = Literal["linear_regression", "random_forest", "gradient_boosting"]
ParamValue = bool | int | float | str | None


class ConfigError(ValueError):
    """The Experiment Config is invalid; the message lists every problem."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Features(_Strict):
    bedrooms_per_room: bool = False


class FixedTuning(_Strict):
    mode: Literal["fixed"] = "fixed"
    # Estimator-level names, e.g. `n_estimators`; the pipeline adds no prefixes to learn.
    params: dict[str, ParamValue] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    @classmethod
    def _single_values(cls, params: Any) -> Any:
        if isinstance(params, dict):
            searched = sorted(k for k, v in params.items() if isinstance(v, list | dict))
            if searched:
                raise ValueError(
                    "fixed mode takes single values; use mode: random to search "
                    f"lists or ranges ({', '.join(searched)})"
                )
        return params

    @property
    def candidate_count(self) -> int:
        return 1

    def candidates(self, defaults: dict[str, Any], random_state: int) -> list[dict[str, Any]]:
        return [{**defaults, **self.params}]


class IntRange(_Strict):
    """Integers from `low` to `high` inclusive, sampled uniformly."""

    type: Literal["int"]
    low: int
    high: int

    @model_validator(mode="after")
    def _ordered(self) -> "IntRange":
        if self.high < self.low:
            raise ValueError("high must be at least low")
        return self

    def distribution(self) -> Any:
        return randint(self.low, self.high + 1)


class FloatRange(_Strict):
    """Floats between `low` and `high`, sampled uniformly or, with `log`, log-uniformly."""

    type: Literal["float"]
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatRange":
        if self.high <= self.low:
            raise ValueError("high must be greater than low")
        if self.log and self.low <= 0:
            raise ValueError("a log range needs low > 0")
        return self

    def distribution(self) -> Any:
        if self.log:
            return loguniform(self.low, self.high)
        return uniform(self.low, self.high - self.low)


Range = Annotated[IntRange | FloatRange, Field(discriminator="type")]
SearchValues = Annotated[list[ParamValue], Field(min_length=1)] | Range


class RandomTuning(_Strict):
    mode: Literal["random"]
    n_iter: int = Field(default=10, ge=1, le=50)
    # Lists are sampled uniformly; ranges map to scipy randint, uniform or loguniform.
    params: dict[str, SearchValues]

    @model_validator(mode="after")
    def _enough_combinations(self) -> "RandomTuning":
        if not self.params:
            raise ValueError("random mode needs at least one param to search")
        if all(isinstance(values, list) for values in self.params.values()):
            combinations = math.prod(len(values) for values in self.params.values())
            if combinations < self.n_iter:
                raise ValueError(
                    f"the lists give only {combinations} distinct Candidates; "
                    f"set n_iter to at most {combinations} or use a range"
                )
        return self

    @property
    def candidate_count(self) -> int:
        return self.n_iter

    def candidates(self, defaults: dict[str, Any], random_state: int) -> list[dict[str, Any]]:
        space = {
            name: values if isinstance(values, list) else values.distribution()
            for name, values in self.params.items()
        }
        return [
            {**defaults, **{name: _native(value) for name, value in params.items()}}
            for params in ParameterSampler(space, self.n_iter, random_state=random_state)
        ]


def _native(value: Any) -> Any:
    """numpy scalars from scipy become plain Python values, for JSON and MLflow params."""
    return value.item() if hasattr(value, "item") else value


class ExperimentConfig(_Strict):
    name: str = Field(min_length=1, max_length=200)
    algorithm: AlgorithmName
    features: Features = Features()
    target_transform: Literal["none", "log1p"] = "none"
    tuning: Annotated[FixedTuning | RandomTuning, Field(discriminator="mode")] = FixedTuning()
    random_state: int = 42

    @field_validator("tuning", mode="before")
    @classmethod
    def _fixed_by_default(cls, tuning: Any) -> Any:
        if isinstance(tuning, dict) and "mode" not in tuning:
            return {"mode": "fixed", **tuning}
        return tuning

    @field_validator("name")
    @classmethod
    def _strip_name(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("name must not be blank")
        return name.strip()

    @model_validator(mode="after")
    def _params_belong_to_the_algorithm(self) -> "ExperimentConfig":
        names = set(self.tuning.params)
        if "random_state" in names:
            raise ValueError("set random_state at the top level of the config, not in params")
        unknown = sorted(names - ALGORITHMS[self.algorithm].param_names())
        if unknown:
            raise ValueError(
                f"unknown hyperparameter(s) for {self.algorithm}: {', '.join(unknown)}"
            )
        return self

    @property
    def candidate_count(self) -> int:
        return self.tuning.candidate_count

    def candidates(self) -> list[dict[str, Any]]:
        """The hyperparameter sets to evaluate: one for fixed mode, `n_iter` for random mode.

        Params the config leaves out keep the algorithm's default fixed values.
        """
        return self.tuning.candidates(ALGORITHMS[self.algorithm].default_params, self.random_state)


def parse_config(raw: dict[str, Any] | ExperimentConfig) -> ExperimentConfig:
    if isinstance(raw, ExperimentConfig):
        return raw
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(p) for p in error['loc']) or 'config'}: {error['msg']}"
            for error in exc.errors()
        ]
        raise ConfigError("invalid Experiment Config:\n  " + "\n  ".join(problems)) from None
