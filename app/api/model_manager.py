"""Holds the Active Model in memory and switches it only after validation (ADR-0001)."""

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from app.db.repositories import Deployment, DeploymentRepository
from app.ml.schema import FEATURES, SAMPLE_ROWS
from app.registry.client import RegistryClient, RegistryModel
from app.registry.model_version import ModelVersionRef

log = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """A Model Version cannot be loaded or does not pass validation; the message says why."""


class DeploymentWriteError(Exception):
    """The Deployment could not be recorded, so the Active Model was not switched."""


@dataclass(frozen=True)
class LoadedModel:
    predictor: RegistryModel
    ref: ModelVersionRef
    run_id: str
    algorithm: str | None
    loaded_at: datetime


@dataclass(frozen=True)
class ModelStatus:
    loaded: LoadedModel | None
    # The Model Version the latest Deployment names: what should be loaded.
    desired: ModelVersionRef | None

    @property
    def in_sync(self) -> bool:
        return self.loaded is not None and self.loaded.ref == self.desired


class ModelManager:
    def __init__(
        self, registry: RegistryClient, deployments: DeploymentRepository, environment: str
    ) -> None:
        self._registry = registry
        self._deployments = deployments
        self._environment = environment
        self._active: LoadedModel | None = None
        # Reentrant, so deploy and reconcile can hold it across load, record and swap.
        self._lock = threading.RLock()

    def current(self) -> LoadedModel | None:
        """The loaded Active Model. Read it once per request: a swap may happen at any time."""
        return self._active

    def status(self) -> ModelStatus:
        latest = self._deployments.latest(self._environment)
        return ModelStatus(loaded=self._active, desired=None if latest is None else latest.ref)

    def history(self, limit: int) -> list[Deployment]:
        """This environment's Deployments, newest first."""
        return self._deployments.history(self._environment, limit)

    def load_and_validate(self, ref: ModelVersionRef) -> LoadedModel:
        """Load a Model Version and check it against the schema. Changes nothing."""
        try:
            predictor = self._registry.load_model(ref)
            metadata = self._registry.get_version_metadata(ref)
        except Exception as exc:
            # Whatever the registry raises, the deploy is refused with its reason.
            raise ModelValidationError(str(exc)) from exc
        _validate(predictor, ref)
        return LoadedModel(
            predictor=predictor,
            ref=ref,
            run_id=metadata.run_id,
            algorithm=metadata.algorithm,
            loaded_at=datetime.now(UTC),
        )

    def swap(self, loaded: LoadedModel) -> None:
        with self._lock:
            self._active = loaded

    def reconcile(self) -> LoadedModel | None:
        """Load the latest Deployment's Model Version, writing nothing.

        Any failure leaves the API running without an Active Model (degraded), not crashed.
        """
        with self._lock:
            try:
                latest = self._deployments.latest(self._environment)
                if latest is None:
                    log.warning("no Deployment in %s; starting degraded", self._environment)
                    return None
                loaded = self.load_and_validate(latest.ref)
            except Exception:
                log.exception("cannot load the Active Model; starting degraded")
                return None
            self.swap(loaded)
            log.info("serving %s", loaded.ref)
            return loaded

    def deploy(self, ref: ModelVersionRef) -> LoadedModel:
        """Validate, record the Deployment, then swap. Writes nothing if `ref` is loaded."""
        with self._lock:
            if self._active is not None and self._active.ref == ref:
                return self._active
            loaded = self.load_and_validate(ref)
            # Record first, then swap: a Model Version is never served without its Deployment.
            try:
                self._deployments.append(self._environment, ref)
            except SQLAlchemyError as exc:
                log.exception("could not record the Deployment of %s", ref)
                raise DeploymentWriteError(
                    f"could not record the Deployment of {ref}; nothing was switched"
                ) from exc
            self.swap(loaded)
            return loaded


def _validate(predictor: RegistryModel, ref: ModelVersionRef) -> None:
    if predictor.input_names != list(FEATURES):
        raise ModelValidationError(
            f"Model Version {ref}: signature columns {predictor.input_names} "
            f"do not equal the schema's features {list(FEATURES)}"
        )
    sample = pd.DataFrame(SAMPLE_ROWS, columns=list(FEATURES))
    try:
        predictions = np.asarray(predictor.predict(sample), dtype="float64")
    except Exception as exc:
        raise ModelValidationError(
            f"Model Version {ref}: predicting the sample rows failed: {exc}"
        ) from exc
    if predictions.shape != (len(sample),):
        raise ModelValidationError(
            f"Model Version {ref}: predictions have shape {predictions.shape}, "
            f"expected ({len(sample)},)"
        )
    if not np.isfinite(predictions).all():
        raise ModelValidationError(
            f"Model Version {ref}: predictions on the sample rows are not all finite"
        )
