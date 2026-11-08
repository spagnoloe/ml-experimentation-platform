"""Consumer routes, versioned under /v1 so the request contract can evolve on its own."""

import json
import logging
import time

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.api.dependencies import Manager
from app.api.schemas import LoadedModelOut, ModelVersionOut
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


class ModelStatusOut(BaseModel):
    loaded: LoadedModelOut | None
    desired: ModelVersionOut | None
    in_sync: bool


class Health(BaseModel):
    status: str
    model_loaded: bool


@router.get("/health")
def health(manager: Manager) -> Health:
    loaded = manager.current() is not None
    return Health(status="ok" if loaded else "degraded", model_loaded=loaded)


@router.get("/v1/model")
def model_status(manager: Manager) -> ModelStatusOut:
    """The loaded Model Version, the one the latest Deployment names, and whether they agree."""
    status = manager.status()
    return ModelStatusOut(
        loaded=None if status.loaded is None else LoadedModelOut.of(status.loaded),
        desired=None if status.desired is None else ModelVersionOut.of(status.desired),
        in_sync=status.in_sync,
    )


@router.post("/v1/predict")
def predict(body: PredictRequest, manager: Manager) -> PredictResponse:
    started = time.perf_counter()
    # Read the reference once, so a concurrent swap cannot mix Model Versions in one response.
    loaded = manager.current()
    if loaded is None:
        raise HTTPException(503, "no Model Version loaded")
    frame = pd.DataFrame([row.model_dump() for row in body.rows], columns=list(FEATURES))
    predictions = loaded.predictor.predict(frame)
    log.info(
        json.dumps(
            {
                "event": "predict",
                "model_name": loaded.ref.model_name,
                "model_version": loaded.ref.model_version,
                "rows": len(frame),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    )
    return PredictResponse(
        predictions=[float(p) for p in predictions],
        unit=TARGET_UNIT,
        model_name=loaded.ref.model_name,
        model_version=loaded.ref.model_version,
    )
