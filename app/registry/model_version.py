"""The exact name of a Model Version: deployments pin these, never aliases (ADR-0001)."""

from dataclasses import dataclass

# Postgres `integer`, the type of `deployments.model_version`.
MAX_MODEL_VERSION = 2**31 - 1


@dataclass(frozen=True)
class ModelVersionRef:
    model_name: str
    model_version: int

    def __str__(self) -> str:
        return f"{self.model_name}/{self.model_version}"
