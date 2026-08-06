"""SQLModel persistence entities."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


class ClientRecord(SQLModel, table=True):
    """An agricultural facility registered with the FL system."""

    __tablename__ = "clients"

    id: str = Field(primary_key=True, max_length=36)
    name: str = Field(index=True, max_length=120)
    description: str = Field(default="", max_length=500)
    status: str = Field(default="registered", index=True, max_length=24)
    partition_id: int | None = Field(default=None)
    num_local_samples: int | None = Field(default=None)
    last_seen_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ExperimentRecord(SQLModel, table=True):
    __tablename__ = "experiments"

    id: str = Field(primary_key=True)
    name: str = Field(index=True, max_length=120)
    status: str = Field(default="draft", index=True, max_length=24)
    execution_mode: str = Field(default="synthetic-smoke", max_length=32)
    algorithm: str = Field(max_length=16)
    partition_kind: str = Field(max_length=16)
    num_clients: int
    num_rounds: int
    local_epochs: int
    learning_rate: float
    batch_size: int
    dirichlet_alpha: float
    proximal_mu: float
    scaffold_server_lr: float = 1.0
    moon_temperature: float = 0.5
    moon_mu: float = 1.0
    seed: int
    result_json: str | None = Field(default=None)
    error_message: str | None = Field(default=None, max_length=2_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ExperimentRoundRecord(SQLModel, table=True):
    """Query-friendly round summary while retaining the complete metric payload."""

    __tablename__ = "experiment_rounds"

    experiment_id: str = Field(
        foreign_key="experiments.id",
        primary_key=True,
        max_length=36,
    )
    round_number: int = Field(primary_key=True, ge=0)
    metrics_json: str
    train_loss: float | None = Field(default=None)
    evaluation_loss: float | None = Field(default=None)
    accuracy: float | None = Field(default=None)
    macro_f1: float | None = Field(default=None)
    harmful_missed_as_healthy_rate: float | None = Field(default=None)
    elapsed_seconds: float | None = Field(default=None)
    bytes_up: int | None = Field(default=None)
    bytes_down: int | None = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now)


class ClientRoundMetricRecord(SQLModel, table=True):
    """Per-client evidence for one train/evaluate phase of a Flower round."""

    __tablename__ = "client_round_metrics"

    experiment_id: str = Field(
        foreign_key="experiments.id",
        primary_key=True,
        max_length=36,
    )
    round_number: int = Field(primary_key=True, ge=1)
    client_id: int = Field(primary_key=True, ge=0)
    phase: str = Field(primary_key=True, max_length=16)
    node_id: str = Field(max_length=32)
    num_examples: int = Field(ge=0)
    metrics_json: str
    payload_download_bytes: int = Field(ge=0)
    payload_upload_bytes: int = Field(ge=0)
    model_download_bytes: int = Field(ge=0)
    model_upload_bytes: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
