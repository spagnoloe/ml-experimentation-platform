from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Engine, create_engine, make_url, text

from app.settings import get_settings

if TYPE_CHECKING:
    from app.db.repositories import JobRepository


def _recreate_database(url: str) -> None:
    """Drop and create the database named in `url`, via the server's `postgres` database."""
    target = make_url(url)
    admin = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{target.database}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def db_engine() -> Engine:
    """An engine on a freshly recreated `app_test` database, once per test session."""
    url = get_settings().test_db_url
    _recreate_database(url)
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def jobs(db_engine) -> "JobRepository":
    """The Training Job repository on `app_test`, with an empty jobs table for each test."""
    from app.db.repositories import JobRepository
    from app.db.session import create_tables

    create_tables(db_engine)
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE training_jobs"))
    return JobRepository(db_engine)


@pytest.fixture(scope="session")
def mlflow_store(tmp_path_factory):
    """MLflow tracking in a temporary local store for the whole session, restored afterwards."""
    import mlflow

    previous = mlflow.get_tracking_uri()
    root = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    # Without this, a sqlite store writes artifacts to ./mlruns in the working directory.
    mlflow.create_experiment(
        get_settings().mlflow_experiment, artifact_location=(root / "artifacts").as_uri()
    )
    yield uri
    mlflow.set_tracking_uri(previous)
