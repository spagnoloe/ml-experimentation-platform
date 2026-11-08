"""What routes receive from the app. Routes reach state only through the ModelManager."""

from typing import Annotated

from fastapi import Depends, Request

from app.api.model_manager import ModelManager


def get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


Manager = Annotated[ModelManager, Depends(get_manager)]
