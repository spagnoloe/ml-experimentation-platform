import pytest
from sqlalchemy import Engine, create_engine, make_url, text

from app.settings import get_settings


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
