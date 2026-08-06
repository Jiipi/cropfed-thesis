"""External API contracts."""

from __future__ import annotations

from datetime import datetime
from math import isclose
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExecutionMode = Literal["synthetic-smoke", "flower"]


class ExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Thí nghiệm FL", min_length=3, max_length=120)
    execution_mode: ExecutionMode = "synthetic-smoke"
    algorithm: Literal["fedavg", "fedprox", "fedbn", "scaffold", "moon"] = "fedavg"
    partition_kind: Literal[
        "iid", "dirichlet", "quantity_skew", "feature_skew"
    ] = "dirichlet"
    num_clients: int = Field(default=4, ge=2, le=20)
    num_rounds: int = Field(default=5, ge=1, le=500)
    local_epochs: int = Field(default=2, ge=1, le=100)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    batch_size: int = Field(default=32, ge=1, le=1_024)
    dirichlet_alpha: float = Field(default=0.5, gt=0, le=100)
    proximal_mu: float = Field(default=0.01, ge=0, le=100)
    scaffold_server_lr: float = Field(default=1.0, gt=0, le=10)
    moon_temperature: float = Field(default=0.5, gt=0, le=10)
    moon_mu: float = Field(default=1.0, ge=0, le=100)
    seed: int = Field(default=2026, ge=0)

    @model_validator(mode="after")
    def validate_flower_profile(self) -> ExperimentCreate:
        if self.execution_mode != "flower":
            return self
        if self.num_clients != 4:
            raise ValueError("Flower MVP experiments require exactly four clients")
        if self.learning_rate > 0.01:
            raise ValueError("Flower learning_rate cannot exceed 0.01")
        if self.num_rounds > 100 or self.local_epochs > 20:
            raise ValueError("Flower round/local epoch limits are 100 and 20")
        if self.batch_size > 512:
            raise ValueError("Flower batch_size cannot exceed 512")
        if self.partition_kind == "dirichlet" and not any(
            isclose(self.dirichlet_alpha, allowed, rel_tol=0.0, abs_tol=1e-9)
            for allowed in (0.5, 0.1)
        ):
            raise ValueError("Flower MVP supports Dirichlet alpha 0.5 or 0.1")
        return self


class ExperimentPublic(BaseModel):
    id: str
    name: str
    status: str
    execution_mode: str
    algorithm: str
    partition_kind: str
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
    result: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ProjectPublic(BaseModel):
    official_title: str
    task_interpretation: str
    primary_dataset: str
    primary_model: str
    canonical_num_clients: int
    mandatory_algorithms: list[str]
    privacy_boundary: str


class ClientCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=120)
    description: str = Field(default="", max_length=500)
    partition_id: int | None = Field(default=None, ge=0, le=19)
    num_local_samples: int | None = Field(default=None, ge=0)


class ClientPublic(BaseModel):
    id: str
    name: str
    description: str
    status: str
    partition_id: int | None
    num_local_samples: int | None
    last_seen_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ClientStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["registered", "connected", "disconnected", "training"] = "connected"


class CheckpointPublic(BaseModel):
    filename: str
    path: str
    size_bytes: int
    model_name: str | None
    model_version: str | None
    experiment_type: str | None
    created_at: str | None
    sha256: str | None
    deployed: bool
    eligible_for_deployment: bool
    validation_error: str | None


class PredictionResponse(BaseModel):
    image_name: str
    crop: str
    predicted_class_id: int
    predicted_label: str
    predicted_group: str
    confidence: float
    model: str
    model_version: str
    checkpoint_format_version: int
    checkpoint_sha256: str
    predictions: list[dict[str, Any]]
    inference_ms: float
    warning: str
    privacy_notice: str
    image_uploaded: bool
