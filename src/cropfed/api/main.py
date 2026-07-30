"""HTTP control plane.

Only the synthetic smoke job runs in FastAPI's lightweight background task.
Real image training is delegated to the Flower runtime and must not run inside a
web worker.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, select

from cropfed.api.auth import Principal, build_auth_dependencies
from cropfed.api.data_profiles import data_profile_payload
from cropfed.api.db import engine, get_session
from cropfed.api.models import (
    ClientRoundMetricRecord,
    ExperimentRecord,
    ExperimentRoundRecord,
)
from cropfed.api.results import (
    client_metric_payload,
    replace_client_history,
    replace_round_history,
    round_payload,
    round_summary,
)
from cropfed.api.schemas import ExperimentCreate, ExperimentPublic, ProjectPublic
from cropfed.api.settings import Settings, settings
from cropfed.config import ExperimentConfig
from cropfed.constants import (
    OFFICIAL_TITLE,
    TOMATO_CLASS_GROUPS,
    TOMATO_CLASSES,
)
from cropfed.simulation import run_synthetic_experiment

SyntheticExecutor = Callable[[ExperimentConfig], dict[str, object]]
SessionDependency = Annotated[Session, Depends(get_session)]


def create_app(
    *,
    database_engine: Engine = engine,
    application_settings: Settings = settings,
    synthetic_executor: SyntheticExecutor = run_synthetic_experiment,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        SQLModel.metadata.create_all(database_engine)
        yield

    def session_provider() -> Generator[Session, None, None]:
        with Session(database_engine) as session:
            yield session

    application = FastAPI(
        title="CropFed Thesis API",
        version="0.1.0",
        description="Control plane for reproducible FL experiments; never uploads raw images.",
        lifespan=lifespan,
    )
    application.dependency_overrides[get_session] = session_provider
    authenticate, require_reader, require_admin = build_auth_dependencies(
        application_settings
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(application_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/health/ready")
    def readiness() -> dict[str, str]:
        try:
            with Session(database_engine) as session:
                session.connection().exec_driver_sql("SELECT 1")
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database is unavailable",
            ) from error
        return {"status": "ok", "database": "ok"}

    @application.get("/api/v1/auth/me")
    def auth_me(
        principal: Principal = Depends(authenticate),  # noqa: B008
    ) -> dict[str, object]:
        return {
            "role": principal.role,
            "authentication_enabled": principal.authentication_enabled,
        }

    @application.get(
        "/api/v1/project",
        response_model=ProjectPublic,
        dependencies=[Depends(require_reader)],
    )
    def project() -> ProjectPublic:
        return ProjectPublic(
            official_title=OFFICIAL_TITLE,
            task_interpretation=(
                "Phát hiện ở mức ảnh: phân loại đa lớp một ảnh lá "
                f"cà chua vào {len(TOMATO_CLASSES)} trạng thái sâu/bệnh/khỏe."
            ),
            primary_dataset="PlantVillage - tomato subset",
            primary_model="MobileNetV2 transfer learning",
            canonical_num_clients=4,
            mandatory_algorithms=[
                "local-only",
                "centralized",
                "FedAvg",
                "FedProx",
            ],
            privacy_boundary=(
                "Ảnh thô ở lại client; server chỉ nhận model update, "
                "số mẫu và metric."
            ),
        )

    @application.get("/api/v1/classes", dependencies=[Depends(require_reader)])
    def classes() -> dict[str, object]:
        return {
            "count": len(TOMATO_CLASSES),
            "items": [
                {
                    "id": class_id,
                    "name": name,
                    "group": TOMATO_CLASS_GROUPS[class_id],
                }
                for class_id, name in enumerate(TOMATO_CLASSES)
            ],
        }

    @application.get(
        "/api/v1/data-profiles", dependencies=[Depends(require_reader)]
    )
    def data_profiles() -> dict[str, object]:
        return data_profile_payload(application_settings)

    @application.post(
        "/api/v1/experiments",
        response_model=ExperimentPublic,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_admin)],
    )
    def create_experiment(
        payload: ExperimentCreate, session: SessionDependency
    ) -> ExperimentPublic:
        record = ExperimentRecord(
            id=str(uuid4()),
            name=payload.name,
            execution_mode=payload.execution_mode,
            algorithm=payload.algorithm,
            partition_kind=payload.partition_kind,
            num_clients=payload.num_clients,
            num_rounds=payload.num_rounds,
            local_epochs=payload.local_epochs,
            learning_rate=payload.learning_rate,
            batch_size=payload.batch_size,
            dirichlet_alpha=payload.dirichlet_alpha,
            proximal_mu=payload.proximal_mu,
            seed=payload.seed,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_public(record)

    @application.get(
        "/api/v1/experiments/{experiment_id}/clients",
        dependencies=[Depends(require_reader)],
    )
    def get_client_metrics(
        experiment_id: str, session: SessionDependency
    ) -> dict[str, object]:
        _require_experiment(session, experiment_id)
        records = session.exec(
            select(ClientRoundMetricRecord)
            .where(ClientRoundMetricRecord.experiment_id == experiment_id)
            .order_by(
                ClientRoundMetricRecord.round_number.asc(),
                ClientRoundMetricRecord.client_id.asc(),
                ClientRoundMetricRecord.phase.asc(),
            )
        ).all()
        return {
            "experiment_id": experiment_id,
            "items": [client_metric_payload(record) for record in records],
        }

    @application.get(
        "/api/v1/experiments",
        response_model=list[ExperimentPublic],
        dependencies=[Depends(require_reader)],
    )
    def list_experiments(session: SessionDependency) -> list[ExperimentPublic]:
        records = session.exec(
            select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())
        ).all()
        return [_to_public(record) for record in records]

    @application.get(
        "/api/v1/experiments/{experiment_id}",
        response_model=ExperimentPublic,
        dependencies=[Depends(require_reader)],
    )
    def get_experiment(
        experiment_id: str, session: SessionDependency
    ) -> ExperimentPublic:
        return _to_public(_require_experiment(session, experiment_id))

    @application.get(
        "/api/v1/experiments/{experiment_id}/rounds",
        dependencies=[Depends(require_reader)],
    )
    def get_rounds(
        experiment_id: str, session: SessionDependency
    ) -> dict[str, object]:
        record = _require_experiment(session, experiment_id)
        result = json.loads(record.result_json) if record.result_json else {}
        round_records = session.exec(
            select(ExperimentRoundRecord)
            .where(ExperimentRoundRecord.experiment_id == experiment_id)
            .order_by(ExperimentRoundRecord.round_number.asc())
        ).all()
        if round_records:
            return {
                "experiment_id": experiment_id,
                "result_kind": result.get("result_kind"),
                "storage": "database",
                "items": [round_payload(item) for item in round_records],
                "summaries": [round_summary(item) for item in round_records],
            }
        return {
            "experiment_id": experiment_id,
            "result_kind": result.get("result_kind"),
            "storage": "result-json-fallback",
            "items": result.get("history", []),
            "summaries": [],
        }

    @application.post(
        "/api/v1/experiments/{experiment_id}/start",
        response_model=ExperimentPublic,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_admin)],
    )
    def start_experiment(
        experiment_id: str,
        background_tasks: BackgroundTasks,
        session: SessionDependency,
    ) -> ExperimentPublic:
        record = _require_experiment(session, experiment_id)
        if record.status not in {"draft", "failed"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"experiment cannot start from status {record.status!r}",
            )
        if (
            record.execution_mode == "flower"
            and not application_settings.flower_worker_enabled
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Flower worker is disabled by server configuration",
            )
        record.status = "queued"
        record.error_message = None
        record.updated_at = datetime.now(UTC)
        session.add(record)
        session.commit()
        session.refresh(record)
        if record.execution_mode == "synthetic-smoke":
            background_tasks.add_task(
                _execute_synthetic_smoke,
                experiment_id,
                database_engine,
                synthetic_executor,
            )
        return _to_public(record)

    return application


def _require_experiment(session: Session, experiment_id: str) -> ExperimentRecord:
    record = session.get(ExperimentRecord, experiment_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="experiment not found",
        )
    return record


def _to_public(record: ExperimentRecord) -> ExperimentPublic:
    return ExperimentPublic(
        id=record.id,
        name=record.name,
        status=record.status,
        execution_mode=record.execution_mode,
        algorithm=record.algorithm,
        partition_kind=record.partition_kind,
        num_clients=record.num_clients,
        num_rounds=record.num_rounds,
        local_epochs=record.local_epochs,
        learning_rate=record.learning_rate,
        batch_size=record.batch_size,
        dirichlet_alpha=record.dirichlet_alpha,
        proximal_mu=record.proximal_mu,
        seed=record.seed,
        result=json.loads(record.result_json) if record.result_json else None,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _execute_synthetic_smoke(
    experiment_id: str,
    database_engine: Engine,
    synthetic_executor: SyntheticExecutor,
) -> None:
    with Session(database_engine) as session:
        record = session.get(ExperimentRecord, experiment_id)
        if record is None:
            return
        try:
            record.status = "running"
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()

            config = ExperimentConfig(
                algorithm=record.algorithm,
                partition_kind=record.partition_kind,
                num_clients=record.num_clients,
                num_rounds=record.num_rounds,
                local_epochs=record.local_epochs,
                learning_rate=record.learning_rate,
                batch_size=record.batch_size,
                dirichlet_alpha=record.dirichlet_alpha,
                proximal_mu=record.proximal_mu,
                seed=record.seed,
            )
            result = synthetic_executor(config)
            record.result_json = json.dumps(result, ensure_ascii=False)
            replace_round_history(session, experiment_id, result)
            replace_client_history(session, experiment_id, result)
            record.status = "completed"
        except Exception as error:  # The failure must be persisted for the dashboard.
            record.status = "failed"
            record.error_message = f"{type(error).__name__}: {error}"[:2_000]
        finally:
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()


app = create_app()
