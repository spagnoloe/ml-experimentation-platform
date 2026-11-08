"""Admin routes, kept apart so production can put them behind auth or an internal network."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import Manager
from app.api.model_manager import DeploymentWriteError, ModelValidationError
from app.api.schemas import LoadedModelOut
from app.registry.model_version import MAX_MODEL_VERSION, ModelVersionRef

router = APIRouter(tags=["admin"])


class DeployRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(min_length=1)
    version: int = Field(ge=1, le=MAX_MODEL_VERSION)


class DeploymentOut(BaseModel):
    id: int
    environment: str
    model_name: str
    model_version: int
    created_at: datetime


@router.post("/deploy")
def deploy(body: DeployRequest, manager: Manager) -> LoadedModelOut:
    """Validate a Model Version, record its Deployment, then make it the Active Model."""
    try:
        loaded = manager.deploy(ModelVersionRef(body.model_name, body.version))
    except ModelValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except DeploymentWriteError as exc:
        raise HTTPException(500, str(exc)) from exc
    return LoadedModelOut.of(loaded)


@router.get("/deployments")
def deployments(manager: Manager, limit: int = Query(50, ge=1, le=500)) -> list[DeploymentOut]:
    """Deployments in this environment, newest first."""
    return [DeploymentOut.model_validate(d, from_attributes=True) for d in manager.history(limit)]
