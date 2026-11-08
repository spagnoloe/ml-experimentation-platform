"""The only code that talks to the MLflow registry. Tests replace it with a fake."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


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
    def load_model(self, name: str, version: int) -> RegistryModel: ...

    def get_version_metadata(self, name: str, version: int) -> VersionMetadata: ...


class _PyfuncModel:
    def __init__(self, model) -> None:
        self._model = model
        schema = model.metadata.get_input_schema()
        self.input_names = None if schema is None else schema.input_names()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(frame))


class MlflowRegistryClient:
    """Loads exact Model Versions (`models:/<name>/<version>`), never aliases (ADR-0001)."""

    def __init__(self, tracking_uri: str) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        self._mlflow = mlflow
        self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def load_model(self, name: str, version: int) -> RegistryModel:
        try:
            return _PyfuncModel(self._mlflow.pyfunc.load_model(f"models:/{name}/{version}"))
        except Exception as exc:
            raise RegistryError(f"cannot load Model Version {name}/{version}: {exc}") from exc

    def get_version_metadata(self, name: str, version: int) -> VersionMetadata:
        try:
            run_id = self._client.get_model_version(name, str(version)).run_id
            algorithm = self._client.get_run(run_id).data.tags.get("algorithm")
        except Exception as exc:
            raise RegistryError(f"cannot read Model Version {name}/{version}: {exc}") from exc
        return VersionMetadata(run_id=run_id, algorithm=algorithm)
