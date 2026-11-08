"""Our tables in the `app` database. MLflow owns its own schema in the `mlflow` database."""

import uuid

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Identity,
    Integer,
    MetaData,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

JOB_STATUSES = ("queued", "running", "completed", "failed")

training_jobs = Table(
    "training_jobs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("name", Text, nullable=False),
    Column("config", JSONB, nullable=False),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("phase", Text),
    Column("progress_done", Integer, nullable=False, server_default="0"),
    Column("progress_total", Integer, nullable=False),
    Column("mlflow_run_id", Text),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint(
        "status IN (" + ", ".join(f"'{s}'" for s in JOB_STATUSES) + ")",
        name="training_jobs_status",
    ),
    CheckConstraint("phase IN ('cv', 'refit', 'logging')", name="training_jobs_phase"),
)

# Append-only: rows are never updated or deleted. The highest id per environment names the
# Active Model (ADR-0001).
deployments = Table(
    "deployments",
    metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("environment", Text, nullable=False),
    Column("model_name", Text, nullable=False),
    Column("model_version", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
