"""The Experiment Config: the data scientist's validated description of what to train."""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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


class ExperimentConfig(_Strict):
    name: str = Field(min_length=1, max_length=200)
    algorithm: AlgorithmName
    features: Features = Features()
    target_transform: Literal["none", "log1p"] = "none"
    tuning: FixedTuning = FixedTuning()
    random_state: int = 42

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

    def candidates(self) -> list[dict[str, Any]]:
        """The hyperparameter sets to evaluate: a fixed config has exactly one Candidate."""
        return [{**ALGORITHMS[self.algorithm].default_params, **self.tuning.params}]


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
