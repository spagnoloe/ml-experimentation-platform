"""Consumer routes, versioned under /v1 so the request contract can evolve on its own."""

import json
import logging
import time
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.api.model_manager import LoadedModel, ModelManager
from app.db.repositories import DeploymentRepository
from app.ml.schema import FEATURES, TARGET_UNIT
from app.settings import get_settings

router = APIRouter()
log = logging.getLogger("app.api.predict")

# Every feature is a required, finite float; ints are accepted, strings and bools are not.
Row = create_model(
    "Row",
    __config__=ConfigDict(extra="forbid", strict=True, allow_inf_nan=False),
    **{name: (float, ...) for name in FEATURES},
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[Row] = Field(min_length=1, max_length=get_settings().max_predict_rows)


class PredictResponse(BaseModel):
    predictions: list[float]
    unit: str
    model_name: str
    model_version: int


class ModelVersionRef(BaseModel):
    model_name: str
    model_version: int


class LoadedModelOut(ModelVersionRef):
    run_id: str
    algorithm: str | None
    loaded_at: datetime

    @classmethod
    def of(cls, loaded: LoadedModel) -> "LoadedModelOut":
        return cls.model_validate(loaded, from_attributes=True)


class ModelStatus(BaseModel):
    loaded: LoadedModelOut | None
    desired: ModelVersionRef | None
    in_sync: bool


class Health(BaseModel):
    status: str
    model_loaded: bool


def _manager(request: Request) -> ModelManager:
    return request.app.state.manager


@router.get("/health")
def health(request: Request) -> Health:
    loaded = _manager(request).current() is not None
    return Health(status="ok" if loaded else "degraded", model_loaded=loaded)


@router.get("/v1/model")
def model(request: Request) -> ModelStatus:
    """The loaded model, the Model Version the latest Deployment names, and whether they agree."""
    loaded = _manager(request).current()
    repository: DeploymentRepository = request.app.state.deployments
    latest = repository.latest(get_settings().environment)
    desired = (
        None
        if latest is None
        else ModelVersionRef(model_name=latest.model_name, model_version=latest.model_version)
    )
    in_sync = (
        loaded is not None
        and desired is not None
        and (loaded.model_name, loaded.model_version) == (desired.model_name, desired.model_version)
    )
    return ModelStatus(
        loaded=None if loaded is None else LoadedModelOut.of(loaded),
        desired=desired,
        in_sync=in_sync,
    )


@router.post("/v1/predict")
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    started = time.perf_counter()
    # Read the model reference once, so a concurrent swap cannot mix Model Versions.
    loaded = _manager(request).current()
    if loaded is None:
        raise HTTPException(503, "no model loaded")
    frame = pd.DataFrame([row.model_dump() for row in body.rows], columns=list(FEATURES))
    predictions = loaded.model.predict(frame)
    log.info(
        json.dumps(
            {
                "event": "predict",
                "model_name": loaded.model_name,
                "model_version": loaded.model_version,
                "rows": len(frame),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    )
    return PredictResponse(
        predictions=[float(p) for p in predictions],
        unit=TARGET_UNIT,
        model_name=loaded.model_name,
        model_version=loaded.model_version,
    )
