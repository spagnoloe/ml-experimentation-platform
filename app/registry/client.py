"""The only code that talks to the MLflow registry. Tests replace it with a fake."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from app.registry.model_version import ModelVersionRef

# Copied from the Run onto each Model Version it is registered as.
_RUN_TAGS = ("job_id", "algorithm", "dataset_sha256")
_RUN_METRICS = ("cv_rmse_mean", "cv_rmse_std", "test_rmse")


class RegistryError(Exception):
    """The registry cannot find, load or register a Model Version; the message says why."""


@dataclass(frozen=True)
class VersionMetadata:
    run_id: str
    algorithm: str | None


@dataclass(frozen=True)
class VersionSummary:
    """A Model Version as the Registry page lists it."""

    ref: ModelVersionRef
    run_id: str
    created_at: datetime
    # The tags copied from the Run at registration (`register_run`).
    tags: dict[str, str]
    # Made from the Best Run, so the Registry page gives it the "best" badge.
    is_best: bool


@dataclass(frozen=True)
class RegisteredModelSummary:
    name: str
    # Newest first.
    versions: list[VersionSummary]


class RegistryModel(Protocol):
    """A loaded Model Version: its signature's input columns and a batch predict."""

    input_names: list[str] | None

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


class RegistryClient(Protocol):
    def load_model(self, ref: ModelVersionRef) -> RegistryModel: ...

    def get_version_metadata(self, ref: ModelVersionRef) -> VersionMetadata: ...


class _PyfuncModel:
    def __init__(self, model: mlflow.pyfunc.PyFuncModel) -> None:
        self._model = model
        schema = model.metadata.get_input_schema()
        self.input_names = None if schema is None else schema.input_names()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(frame))


class MlflowRegistryClient:
    """Loads exact Model Versions (`models:/<name>/<version>`), never aliases (ADR-0001)."""

    def __init__(self, tracking_uri: str) -> None:
        self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def load_model(self, ref: ModelVersionRef) -> RegistryModel:
        uri = f"models:/{ref.model_name}/{ref.model_version}"
        try:
            return _PyfuncModel(mlflow.pyfunc.load_model(uri))
        except Exception as exc:
            raise RegistryError(f"cannot load Model Version {ref}: {exc}") from exc

    def get_version_metadata(self, ref: ModelVersionRef) -> VersionMetadata:
        try:
            version = self._client.get_model_version(ref.model_name, str(ref.model_version))
            algorithm = self._client.get_run(version.run_id).data.tags.get("algorithm")
        except Exception as exc:
            raise RegistryError(f"cannot read Model Version {ref}: {exc}") from exc
        return VersionMetadata(run_id=version.run_id, algorithm=algorithm)

    def register_run(self, run_id: str, model_name: str) -> ModelVersionRef:
        """Register a completed Run's pipeline as the next Model Version of `model_name`.

        A Run is registered at most once per name. The version is tagged so the registry needs
        no Run lookups. Nothing is deployed (ADR-0001).
        """
        if existing := self.version_of_run(run_id, model_name):
            raise RegistryError(f"Run {run_id} is already registered as {existing}")
        try:
            run = self._client.get_run(run_id)
        except Exception as exc:
            raise RegistryError(f"cannot read Run {run_id}: {exc}") from exc
        tags = {name: run.data.tags.get(name) for name in _RUN_TAGS}
        tags |= {name: run.data.metrics.get(name) for name in _RUN_METRICS}
        if missing := [name for name, value in tags.items() if value is None]:
            raise RegistryError(f"Run {run_id} has no {', '.join(missing)}; is it completed?")
        # MLflow 3 logs the pipeline as a Logged Model output of the Run, not a run artifact.
        outputs = run.outputs.model_outputs if run.outputs else []
        if len(outputs) != 1:
            raise RegistryError(f"Run {run_id} has {len(outputs)} logged models, expected 1")
        try:
            version = mlflow.register_model(
                f"models:/{outputs[0].model_id}",
                model_name,
                tags={name: str(value) for name, value in tags.items()},
                await_registration_for=60,
            )
        except Exception as exc:
            raise RegistryError(f"cannot register Run {run_id} as {model_name}: {exc}") from exc
        return ModelVersionRef(model_name, int(version.version))

    def version_of_run(self, run_id: str, model_name: str) -> ModelVersionRef | None:
        """The Model Version of `model_name` made from this Run, if it is registered."""
        try:
            found = self._client.search_model_versions(
                f"name = '{model_name}' and run_id = '{run_id}'"
            )
        except Exception as exc:
            raise RegistryError(f"cannot search Model Versions of {model_name}: {exc}") from exc
        # Only the oldest counts, should versions ever be registered outside this client.
        oldest = min((int(v.version) for v in found), default=None)
        return None if oldest is None else ModelVersionRef(model_name, oldest)

    def registered_models(self, *, best_run_id: str | None) -> list[RegisteredModelSummary]:
        """Every Registered Model by name, with its Model Versions newest first.

        A version made from `best_run_id` (the Best Run, if there is one) is marked `is_best`.
        """
        try:
            models = self._client.search_registered_models(order_by=["name ASC"])
            versions = self._client.search_model_versions()
        except Exception as exc:
            raise RegistryError(f"cannot list Registered Models: {exc}") from exc
        by_name: dict[str, list[VersionSummary]] = {model.name: [] for model in models}
        for version in versions:
            by_name.setdefault(version.name, []).append(
                VersionSummary(
                    ref=ModelVersionRef(version.name, int(version.version)),
                    run_id=version.run_id,
                    created_at=datetime.fromtimestamp(version.creation_timestamp / 1000, UTC),
                    tags=dict(version.tags),
                    is_best=best_run_id is not None and version.run_id == best_run_id,
                )
            )
        return [
            RegisteredModelSummary(name, sorted(found, key=lambda v: -v.ref.model_version))
            for name, found in sorted(by_name.items())
        ]
