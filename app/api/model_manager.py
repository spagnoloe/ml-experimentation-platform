"""Holds the Active Model in memory and switches it only after validation (ADR-0001)."""

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from app.db.repositories import DeploymentRepository
from app.ml.schema import FEATURES, SAMPLE_ROWS
from app.registry.client import RegistryClient, RegistryError, RegistryModel

log = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """A Model Version cannot be loaded or does not pass validation; the message says why."""


class DeploymentWriteError(Exception):
    """The Deployment could not be recorded, so the Active Model was not switched."""


@dataclass(frozen=True)
class LoadedModel:
    model: RegistryModel
    model_name: str
    model_version: int
    run_id: str
    algorithm: str | None
    loaded_at: datetime


class ModelManager:
    def __init__(
        self, registry: RegistryClient, deployments: DeploymentRepository, environment: str
    ) -> None:
        self._registry = registry
        self._deployments = deployments
        self._environment = environment
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()

    def current(self) -> LoadedModel | None:
        return self._current

    def load_and_validate(self, name: str, version: int) -> LoadedModel:
        """Load a Model Version and check it against the schema. Changes nothing."""
        try:
            model = self._registry.load_model(name, version)
            metadata = self._registry.get_version_metadata(name, version)
        except RegistryError as exc:
            raise ModelValidationError(str(exc)) from exc
        _validate(model, f"{name}/{version}")
        return LoadedModel(
            model=model,
            model_name=name,
            model_version=version,
            run_id=metadata.run_id,
            algorithm=metadata.algorithm,
            loaded_at=datetime.now(UTC),
        )

    def swap(self, loaded: LoadedModel) -> None:
        self._current = loaded

    def reconcile(self) -> LoadedModel | None:
        """Load the latest Deployment's Model Version, writing nothing. None leaves no model."""
        with self._lock:
            latest = self._deployments.latest(self._environment)
            if latest is None:
                log.warning("no Deployment in %s; starting without a model", self._environment)
                return None
            try:
                loaded = self.load_and_validate(latest.model_name, latest.model_version)
            except ModelValidationError:
                log.exception(
                    "cannot load the Active Model %s/%s; starting without a model",
                    latest.model_name,
                    latest.model_version,
                )
                return None
            self.swap(loaded)
            log.info("serving %s/%s", loaded.model_name, loaded.model_version)
            return loaded

    def deploy(self, name: str, version: int) -> LoadedModel:
        with self._lock:
            current = self._current
            if current and (current.model_name, current.model_version) == (name, version):
                return current
            loaded = self.load_and_validate(name, version)
            # Record first, then swap: a model is never served without its Deployment.
            try:
                self._deployments.append(self._environment, name, version)
            except SQLAlchemyError as exc:
                log.exception("could not record the Deployment of %s/%s", name, version)
                raise DeploymentWriteError(
                    f"could not record the Deployment of {name}/{version}; nothing was switched"
                ) from exc
            self.swap(loaded)
            return loaded


def _validate(model: RegistryModel, label: str) -> None:
    if model.input_names != list(FEATURES):
        raise ModelValidationError(
            f"Model Version {label}: signature columns {model.input_names} "
            f"do not equal the schema's features {list(FEATURES)}"
        )
    sample = pd.DataFrame(SAMPLE_ROWS, columns=list(FEATURES))
    try:
        predictions = np.asarray(model.predict(sample), dtype="float64")
    except Exception as exc:
        raise ModelValidationError(
            f"Model Version {label}: predicting the sample rows failed: {exc}"
        ) from exc
    if predictions.shape != (len(sample),):
        raise ModelValidationError(
            f"Model Version {label}: predictions have shape {predictions.shape}, "
            f"expected ({len(sample)},)"
        )
    if not np.isfinite(predictions).all():
        raise ModelValidationError(
            f"Model Version {label}: predictions on the sample rows are not all finite"
        )
