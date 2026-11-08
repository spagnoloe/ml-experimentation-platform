"""Engine and table creation. Tables come from `create_all`; production would use Alembic."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from app.db.tables import metadata
from app.settings import get_settings


def create_tables(engine: Engine) -> None:
    metadata.create_all(engine)


@lru_cache
def get_engine() -> Engine:
    """The engine on the `app` database, with our tables created."""
    engine = create_engine(get_settings().app_db_url, pool_pre_ping=True)
    create_tables(engine)
    return engine
