"""Admin routes, kept apart so production can put them behind auth or an internal network."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.model_manager import DeploymentWriteError, ModelManager, ModelValidationError
from app.api.routes_public import LoadedModelOut
from app.db.repositories import DeploymentRepository
from app.settings import get_settings

router = APIRouter(tags=["admin"])


class DeployRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(min_length=1)
    version: int = Field(ge=1)


def _manager(request: Request) -> ModelManager:
    return request.app.state.manager


@router.post("/deploy")
def deploy(body: DeployRequest, request: Request) -> LoadedModelOut:
    """Validate a Model Version, record its Deployment, then make it the Active Model."""
    try:
        loaded = _manager(request).deploy(body.model_name, body.version)
    except ModelValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except DeploymentWriteError as exc:
        raise HTTPException(500, str(exc)) from exc
    return LoadedModelOut.of(loaded)


class DeploymentOut(BaseModel):
    id: int
    environment: str
    model_name: str
    model_version: int
    created_at: datetime


@router.get("/deployments")
def deployments(request: Request, limit: int = Query(50, ge=1, le=500)) -> list[DeploymentOut]:
    """Deployments in this environment, newest first."""
    repository: DeploymentRepository = request.app.state.deployments
    history = repository.history(get_settings().environment, limit=limit)
    return [DeploymentOut.model_validate(d, from_attributes=True) for d in history]
