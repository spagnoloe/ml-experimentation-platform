"""The only code that talks to the MLflow registry. Tests replace it with a fake."""

from dataclasses import dataclass
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from app.registry.model_version import ModelVersionRef


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
