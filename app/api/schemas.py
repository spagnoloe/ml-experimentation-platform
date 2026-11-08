"""Response bodies shared by the consumer and admin routers."""

from datetime import datetime

from pydantic import BaseModel

from app.api.model_manager import LoadedModel
from app.registry.model_version import ModelVersionRef


class ModelVersionOut(BaseModel):
    model_name: str
    model_version: int

    @classmethod
    def of(cls, ref: ModelVersionRef) -> "ModelVersionOut":
        return cls(model_name=ref.model_name, model_version=ref.model_version)


class LoadedModelOut(ModelVersionOut):
    run_id: str
    algorithm: str | None
    loaded_at: datetime

    @classmethod
    def of(cls, loaded: LoadedModel) -> "LoadedModelOut":
        return cls(
            model_name=loaded.ref.model_name,
            model_version=loaded.ref.model_version,
            run_id=loaded.run_id,
            algorithm=loaded.algorithm,
            loaded_at=loaded.loaded_at,
        )
