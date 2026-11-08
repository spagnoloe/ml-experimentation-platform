"""Settings from environment variables. Defaults suit a host-side dev loop against Compose."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_path: Path = _REPO_ROOT / "data" / "training_data.csv"

    app_db_url: str = "postgresql+psycopg://platform:platform@localhost:5432/app"
    test_db_url: str = "postgresql+psycopg://platform:platform@localhost:5432/app_test"
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "california-housing"
    api_url: str = "http://localhost:8000"

    # Set in the image at build time; on the host it is read from git instead.
    git_sha: str | None = None

    # Evaluation protocol: fixed for every Training Job so all Runs are comparable.
    split_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    cv_seed: int = 42

    # Limits.
    worker_poll_seconds: float = 2.0
    max_n_iter: int = 50
    max_predict_rows: int = 1000


@lru_cache
def get_settings() -> Settings:
    return Settings()
