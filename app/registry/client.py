"""The only code that talks to the MLflow registry. Tests replace it with a fake."""

from dataclasses import dataclass
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
    """A Model Version cannot be found or loaded."""


@dataclass(frozen=True)
class VersionMetadata:
    run_id: str
    algorithm: str | None


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

        The version is tagged so the registry needs no Run lookups. Nothing is deployed.
        """
        run = self._client.get_run(run_id)
        tags = {name: run.data.tags[name] for name in _RUN_TAGS if name in run.data.tags}
        tags |= {
            name: str(run.data.metrics[name]) for name in _RUN_METRICS if name in run.data.metrics
        }
        # MLflow 3 logs the pipeline as a Logged Model output of the Run, not a run artifact.
        outputs = run.outputs.model_outputs if run.outputs else []
        if len(outputs) != 1:
            raise RegistryError(f"Run {run_id} has {len(outputs)} logged models, expected 1")
        version = mlflow.register_model(
            f"models:/{outputs[0].model_id}", model_name, tags=tags, await_registration_for=60
        )
        return ModelVersionRef(model_name, int(version.version))

    def versions_of_run(self, run_id: str, model_name: str) -> list[ModelVersionRef]:
        """The Model Versions of `model_name` made from this Run, oldest first."""
        found = self._client.search_model_versions(f"name = '{model_name}' and run_id = '{run_id}'")
        return [ModelVersionRef(model_name, n) for n in sorted(int(v.version) for v in found)]
