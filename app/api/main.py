"""The serving API. Run with `uvicorn app.api.main:app --workers 1`."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Engine

from app.api import routes_admin, routes_public
from app.api.model_manager import ModelManager
from app.db.repositories import DeploymentRepository
from app.registry.client import RegistryClient
from app.settings import get_settings


def create_app(*, registry: RegistryClient | None = None, engine: Engine | None = None) -> FastAPI:
    """Build the API. Without arguments it uses MLflow and the `app` database from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Our loggers to stdout next to uvicorn's; a no-op if logging is already configured.
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        settings = get_settings()
        if registry is None:
            from app.registry.client import MlflowRegistryClient

            app_registry = MlflowRegistryClient(settings.mlflow_tracking_uri)
        else:
            app_registry = registry
        if engine is None:
            from app.db.session import get_engine

            app_engine = get_engine()
        else:
            app_engine = engine
        app.state.deployments = DeploymentRepository(app_engine)
        app.state.manager = ModelManager(app_registry, app.state.deployments, settings.environment)
        app.state.manager.reconcile()
        yield

    app = FastAPI(title="California house values", lifespan=lifespan)
    app.include_router(routes_public.router)
    app.include_router(routes_admin.router)
    return app


app = create_app()
