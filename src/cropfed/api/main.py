"""HTTP control plane.

Only the synthetic smoke job runs in FastAPI's lightweight background task.
Real image training is delegated to the Flower runtime and must not run inside a
web worker.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import tempfile
import warnings
from collections.abc import Callable, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, select

from cropfed.api.auth import Principal, build_auth_dependencies
from cropfed.api.data_profiles import data_profile_payload
from cropfed.api.db import engine, get_session
from cropfed.api.models import (
    ClientRecord,
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
from cropfed.api.schemas import (
    CheckpointPublic,
    ClientCreate,
    ClientPublic,
    ClientStatusUpdate,
    ExperimentCreate,
    ExperimentPublic,
    PredictionResponse,
    ProjectPublic,
)
from cropfed.api.settings import Settings, settings
from cropfed.config import ExperimentConfig
from cropfed.constants import OFFICIAL_TITLE, taxonomy_from_scope
from cropfed.ml.metrics import client_fairness, gap_vs_centralized
from cropfed.simulation import run_synthetic_experiment

SyntheticExecutor = Callable[[ExperimentConfig], dict[str, object]]
PredictionExecutor = Callable[..., dict[str, object]]
SessionDependency = Annotated[Session, Depends(get_session)]


def create_app(
    *,
    database_engine: Engine = engine,
    application_settings: Settings = settings,
    synthetic_executor: SyntheticExecutor = run_synthetic_experiment,
    prediction_executor: PredictionExecutor | None = None,
    initialize_schema: bool = True,
) -> FastAPI:
    taxonomy = taxonomy_from_scope(application_settings.taxonomy_scope)
    if initialize_schema:
        SQLModel.metadata.create_all(database_engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    def session_provider() -> Generator[Session, None, None]:
        with Session(database_engine) as session:
            yield session

    application = FastAPI(
        title="CropFed Thesis API",
        version="0.1.0",
        description=(
            "Control plane for reproducible FL experiments. FL training never uploads "
            "raw images; the separate demo inference endpoint accepts voluntary uploads."
        ),
        lifespan=lifespan,
    )
    application.dependency_overrides[get_session] = session_provider
    authenticate, require_reader, require_admin = build_auth_dependencies(
        application_settings
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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
                "Phát hiện ở mức ảnh: phân loại đa lớp một ảnh lá cây vào "
                f"{len(taxonomy.class_names)} trạng thái cây/sâu/bệnh/khỏe."
            ),
            primary_dataset=(
                "PlantVillage - full raw/color (38 classes)"
                if taxonomy.scope == "plantvillage-full"
                else "PlantVillage - tomato subset"
            ),
            primary_model=application_settings.flower_model_name,
            canonical_num_clients=4,
            mandatory_algorithms=[
                "local-only",
                "centralized",
                "FedAvg",
                "FedProx",
                "FedBN",
                "SCAFFOLD",
                "MOON",
            ],
            privacy_boundary=(
                "Ảnh thô ở lại client; server chỉ nhận model update, "
                "số mẫu và metric."
            ),
        )

    @application.get("/api/v1/classes", dependencies=[Depends(require_reader)])
    def classes() -> dict[str, object]:
        return {
            "count": len(taxonomy.class_names),
            "taxonomy_scope": taxonomy.scope,
            "items": [
                {
                    "id": class_id,
                    "name": name,
                    "group": taxonomy.class_groups[class_id],
                }
                for class_id, name in enumerate(taxonomy.class_names)
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
            scaffold_server_lr=payload.scaffold_server_lr,
            moon_temperature=payload.moon_temperature,
            moon_mu=payload.moon_mu,
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
        "/api/v1/experiments/compare",
        dependencies=[Depends(require_reader)],
    )
    def compare_experiments(
        session: SessionDependency,
        ids: Annotated[list[str], Query()] = (),
    ) -> dict[str, object]:
        if len(ids) < 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="at least two experiment IDs are required",
            )
        if len(ids) > 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="maximum 10 experiments can be compared",
            )
        if len(set(ids)) != len(ids):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="experiment IDs must be unique",
            )
        missing_ids = [
            experiment_id
            for experiment_id in ids
            if session.get(ExperimentRecord, experiment_id) is None
        ]
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "experiments not found", "ids": missing_ids},
            )
        items = []
        baseline = _load_centralized_baseline(application_settings)
        for experiment_id in ids:
            record = session.get(ExperimentRecord, experiment_id)
            assert record is not None
            result_payload = (
                json.loads(record.result_json) if record.result_json else {}
            )
            rounds = session.exec(
                select(ExperimentRoundRecord)
                .where(ExperimentRoundRecord.experiment_id == experiment_id)
                .order_by(ExperimentRoundRecord.round_number.asc())
            ).all()
            client_records = session.exec(
                select(ClientRoundMetricRecord)
                .where(ClientRoundMetricRecord.experiment_id == experiment_id)
            ).all()
            selection = result_payload.get("selection", {})
            selected_round = (
                int(selection["best_round"])
                if isinstance(selection, dict)
                and selection.get("best_round") is not None
                else max((r.round_number for r in rounds), default=0)
            )
            client_f1s = [
                float(json.loads(cr.metrics_json).get("eval_macro_f1", 0))
                for cr in client_records
                if cr.phase == "evaluate" and cr.round_number == selected_round
            ]
            client_examples = [
                int(cr.num_examples)
                for cr in client_records
                if cr.phase == "evaluate" and cr.round_number == selected_round
            ]
            fairness = (
                client_fairness(
                    client_f1s,
                    num_examples=(
                        client_examples if sum(client_examples) > 0 else None
                    ),
                )
                if len(client_f1s) >= 2
                else None
            )
            selected_metrics = next(
                (row for row in rounds if row.round_number == selected_round),
                rounds[-1] if rounds else None,
            )
            global_test = result_payload.get("global_test", {})
            if not isinstance(global_test, dict):
                global_test = {}
            final_accuracy = global_test.get(
                "global_test_accuracy",
                selected_metrics.accuracy if selected_metrics else None,
            )
            final_macro_f1 = global_test.get(
                "global_test_macro_f1",
                selected_metrics.macro_f1 if selected_metrics else None,
            )
            # Only compare against a baseline trained with the same seed: a
            # different seed makes the gap partly a seed difference, which is
            # exactly the confusion §8's core result cannot afford.
            gap = (
                {
                    "accuracy": gap_vs_centralized(
                        final_accuracy, baseline["accuracy"]
                    ),
                    "macro_f1": gap_vs_centralized(
                        final_macro_f1, baseline["macro_f1"]
                    ),
                }
                if baseline is not None and baseline["seed"] == record.seed
                else None
            )
            items.append({
                "id": record.id,
                "name": record.name,
                "status": record.status,
                "algorithm": record.algorithm,
                "partition_kind": record.partition_kind,
                "dirichlet_alpha": record.dirichlet_alpha,
                "num_clients": record.num_clients,
                "num_rounds": record.num_rounds,
                "seed": record.seed,
                "execution_mode": record.execution_mode,
                "selected_round": selected_round or None,
                "final_accuracy": final_accuracy,
                "final_macro_f1": final_macro_f1,
                "final_harmful_rate": (
                    global_test.get("global_test_harmful_missed_as_healthy_rate")
                    if global_test
                    else selected_metrics.harmful_missed_as_healthy_rate
                    if selected_metrics
                    else None
                ),
                "total_bytes_up": sum(r.bytes_up or 0 for r in rounds),
                "total_bytes_down": sum(r.bytes_down or 0 for r in rounds),
                "total_elapsed": sum(r.elapsed_seconds or 0 for r in rounds),
                "client_f1s": client_f1s if client_f1s else None,
                "worst_client_f1": min(client_f1s) if client_f1s else None,
                "mean_client_f1": sum(client_f1s) / len(client_f1s) if client_f1s else None,
                "fairness": fairness,
                "gap_vs_centralized": gap,
                "round_history": [
                    {
                        "round": r.round_number,
                        "accuracy": r.accuracy,
                        "macro_f1": r.macro_f1,
                        "train_loss": r.train_loss,
                    }
                    for r in rounds
                ],
            })
        return {
            "items": items,
            "centralized_baseline": baseline,
        }

    @application.get(
        "/api/v1/experiments/export-csv",
        dependencies=[Depends(require_reader)],
    )
    def export_csv(
        session: SessionDependency,
        ids: Annotated[list[str], Query()] = (),
    ) -> StreamingResponse:
        if not ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="at least one experiment ID is required",
            )
        if len(ids) > 100 or len(set(ids)) != len(ids):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="provide between 1 and 100 unique experiment IDs",
            )
        missing_ids = [
            experiment_id
            for experiment_id in ids
            if session.get(ExperimentRecord, experiment_id) is None
        ]
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "experiments not found", "ids": missing_ids},
            )
        fieldnames = [
            "id",
            "name",
            "algorithm",
            "partition_kind",
            "dirichlet_alpha",
            "num_clients",
            "num_rounds",
            "seed",
            "status",
            "execution_mode",
            "export_type",
            "research_validation_status",
            "accuracy",
            "macro_f1",
            "harmful_missed_rate",
            "total_elapsed_seconds",
            "total_bytes_up",
            "total_bytes_down",
        ]
        buffer = io.StringIO()
        buffer.write("\ufeff")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for experiment_id in ids:
            record = session.get(ExperimentRecord, experiment_id)
            assert record is not None
            res = json.loads(record.result_json) if record.result_json else {}
            is_valid_research = res.get("research_result_valid") is True
            rounds = session.exec(
                select(ExperimentRoundRecord)
                .where(ExperimentRoundRecord.experiment_id == experiment_id)
                .order_by(ExperimentRoundRecord.round_number.asc())
            ).all()
            selection = res.get("selection", {})
            selected_round = (
                int(selection["best_round"])
                if isinstance(selection, dict)
                and selection.get("best_round") is not None
                else max((row.round_number for row in rounds), default=0)
            )
            selected = next(
                (row for row in rounds if row.round_number == selected_round),
                rounds[-1] if rounds else None,
            )
            global_test = res.get("global_test", {})
            if not isinstance(global_test, dict):
                global_test = {}
            writer.writerow(_csv_safe_row({
                "id": record.id,
                "name": record.name,
                "algorithm": record.algorithm,
                "partition_kind": record.partition_kind,
                "dirichlet_alpha": record.dirichlet_alpha,
                "num_clients": record.num_clients,
                "num_rounds": record.num_rounds,
                "seed": record.seed,
                "status": record.status,
                "execution_mode": record.execution_mode,
                "export_type": "operational_debug_export",
                "research_validation_status": (
                    "validated_research_candidate"
                    if is_valid_research
                    else "debug_or_synthetic"
                ),
                "accuracy": global_test.get(
                    "global_test_accuracy", selected.accuracy if selected else ""
                ),
                "macro_f1": global_test.get(
                    "global_test_macro_f1", selected.macro_f1 if selected else ""
                ),
                "harmful_missed_rate": global_test.get(
                    "global_test_harmful_missed_as_healthy_rate",
                    selected.harmful_missed_as_healthy_rate if selected else "",
                ),
                "total_elapsed_seconds": sum(r.elapsed_seconds or 0 for r in rounds),
                "total_bytes_up": sum(r.bytes_up or 0 for r in rounds),
                "total_bytes_down": sum(r.bytes_down or 0 for r in rounds),
            }))
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cropfed_experiments.csv"},
        )

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

    @application.post(
        "/api/v1/clients",
        response_model=ClientPublic,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_admin)],
    )
    def create_client(
        payload: ClientCreate, session: SessionDependency
    ) -> ClientPublic:
        record = ClientRecord(
            id=str(uuid4()),
            name=payload.name,
            description=payload.description,
            partition_id=payload.partition_id,
            num_local_samples=payload.num_local_samples,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_client_public(record)

    @application.get(
        "/api/v1/clients",
        response_model=list[ClientPublic],
        dependencies=[Depends(require_reader)],
    )
    def list_clients(session: SessionDependency) -> list[ClientPublic]:
        records = session.exec(
            select(ClientRecord).order_by(ClientRecord.created_at.asc())
        ).all()
        return [_to_client_public(record) for record in records]

    @application.patch(
        "/api/v1/clients/{client_id}/status",
        response_model=ClientPublic,
        dependencies=[Depends(require_admin)],
    )
    def update_client_status(
        client_id: str,
        payload: ClientStatusUpdate,
        session: SessionDependency,
    ) -> ClientPublic:
        record = session.get(ClientRecord, client_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="client not found"
            )
        record.status = payload.status
        if payload.status == "connected":
            record.last_seen_at = datetime.now(UTC)
        record.updated_at = datetime.now(UTC)
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_client_public(record)

    @application.post(
        "/api/v1/predict",
        response_model=PredictionResponse,
        dependencies=[Depends(require_reader)],
    )
    async def predict_image_endpoint(
        image: UploadFile = File(...),  # noqa: B008
        checkpoint_name: Annotated[str | None, Query()] = None,
    ) -> PredictionResponse:
        """Voluntary image upload for demo inference — not part of FL training."""

        if checkpoint_name:
            root = application_settings.checkpoint_dir.expanduser().resolve()
            candidate = (root / checkpoint_name).expanduser().resolve()
            if (
                not candidate.is_relative_to(root)
                or not candidate.is_file()
                or candidate.suffix.lower() != ".pt"
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="requested checkpoint was not found inside checkpoint_dir",
                )
            checkpoint_path = candidate
            checkpoint_sha256 = _sha256_file(candidate)
        else:
            try:
                checkpoint_path, checkpoint_sha256 = _resolve_deployed_checkpoint(
                    application_settings
                )
            except (FileNotFoundError, ValueError) as error:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="deployed checkpoint is not configured or failed verification",
                ) from error
        suffix = Path(image.filename or "upload.jpg").suffix.lower()
        max_bytes = 10 * 1024 * 1024
        content_bytes = bytearray()
        chunk_size = 1024 * 64
        while True:
            chunk = await image.read(chunk_size)
            if not chunk:
                break
            content_bytes.extend(chunk)
            if len(content_bytes) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="image exceeds 10 MB limit",
                )
        try:
            _validate_uploaded_image(
                bytes(content_bytes),
                suffix=suffix,
                content_type=image.content_type,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content_bytes)
            tmp_path = Path(tmp.name)

        try:
            import anyio

            if prediction_executor is None:
                from cropfed.ml.inference import predict_image as predictor
            else:
                predictor = prediction_executor

            try:
                result = await anyio.to_thread.run_sync(
                    partial(
                        predictor,
                        checkpoint_path=checkpoint_path,
                        image_path=tmp_path,
                        model_name=None,
                    )
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="deployed checkpoint could not perform inference",
                ) from error
            result["image_name"] = Path(image.filename or "upload").name
            result["image_uploaded"] = True
            result["checkpoint_sha256"] = checkpoint_sha256
            result["privacy_notice"] = (
                "Ảnh này được gửi tự nguyện để demo dự đoán trên server. "
                "Luồng này hoàn toàn tách biệt với dữ liệu huấn luyện riêng tư "
                "của các cơ sở nông nghiệp. Ảnh sẽ bị xóa ngay sau khi xử lý."
            )
            return PredictionResponse.model_validate(result)
        finally:
            tmp_path.unlink(missing_ok=True)

    @application.get(
        "/api/v1/checkpoints",
        response_model=list[CheckpointPublic],
        dependencies=[Depends(require_reader)],
    )
    def list_checkpoints() -> list[CheckpointPublic]:
        checkpoint_dir = application_settings.checkpoint_dir.expanduser().resolve()
        if not checkpoint_dir.is_dir():
            return []
        try:
            deployed_path, deployed_sha256 = _resolve_deployed_checkpoint(
                application_settings
            )
        except (FileNotFoundError, ValueError):
            deployed_path, deployed_sha256 = None, None
        results: list[CheckpointPublic] = []
        candidates = sorted(
            checkpoint_dir.rglob("*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:20]
        for path in candidates:
            info = _checkpoint_info(
                path,
                checkpoint_dir,
                deployed_path,
                deployed_sha256,
                taxonomy.class_names,
            )
            results.append(info)
        return results

    return application


def _csv_safe_row(values: dict[str, object]) -> dict[str, object]:
    """Prevent spreadsheet programs from evaluating user-controlled cells."""

    return {
        key: f"'{value}"
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
        else value
        for key, value in values.items()
    }


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
        scaffold_server_lr=record.scaffold_server_lr,
        moon_temperature=record.moon_temperature,
        moon_mu=record.moon_mu,
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
                scaffold_server_lr=record.scaffold_server_lr,
                moon_temperature=record.moon_temperature,
                moon_mu=record.moon_mu,
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


app = create_app(initialize_schema=False)


def _to_client_public(record: ClientRecord) -> ClientPublic:
    return ClientPublic(
        id=record.id,
        name=record.name,
        description=record.description,
        status=record.status,
        partition_id=record.partition_id,
        num_local_samples=record.num_local_samples,
        last_seen_at=record.last_seen_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _load_centralized_baseline(
    application_settings: Settings,
) -> dict[str, object] | None:
    """Read the configured centralized baseline, or return ``None``.

    Only a *research-valid* centralized result is accepted. A pilot artifact
    carries ``research_result_valid=false`` (D-028) and comparing against it
    would put a number the exporter refuses to publish onto the dashboard as
    the thesis's headline result.

    Failures are absorbed rather than raised: the gap is an extra column, and a
    missing or malformed baseline file must not take the comparison view down.
    """

    configured = application_settings.centralized_baseline_result
    if configured is None:
        return None
    path = configured.expanduser()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("experiment_type") != "centralized":
        return None
    if payload.get("research_result_valid") is not True:
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    # Every Flower experiment this server runs uses one configured backbone, so
    # a baseline trained with a different one would turn an architecture
    # difference into a reported federation cost.
    if payload.get("model") != application_settings.flower_model_name:
        return None
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "model": payload.get("model"),
        "seed": payload.get("seed"),
    }


def _resolve_deployed_checkpoint(
    application_settings: Settings,
) -> tuple[Path, str]:
    configured = application_settings.deployed_checkpoint
    expected_sha256 = application_settings.deployed_checkpoint_sha256
    if configured is None or expected_sha256 is None:
        raise FileNotFoundError("no deployed checkpoint is configured")
    root = application_settings.checkpoint_dir.expanduser().resolve()
    candidate = configured if configured.is_absolute() else root / configured
    resolved = candidate.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("deployed checkpoint must remain inside checkpoint_dir")
    if not resolved.is_file() or resolved.suffix.lower() != ".pt":
        raise FileNotFoundError("deployed checkpoint does not exist")
    actual_sha256 = _sha256_file(resolved)
    if not hmac.compare_digest(actual_sha256, expected_sha256.lower()):
        raise ValueError("deployed checkpoint SHA-256 mismatch")
    return resolved, actual_sha256


def _validate_uploaded_image(
    content: bytes,
    *,
    suffix: str,
    content_type: str | None,
) -> None:
    allowed = {
        ".jpg": ("JPEG", "image/jpeg"),
        ".jpeg": ("JPEG", "image/jpeg"),
        ".png": ("PNG", "image/png"),
        ".bmp": ("BMP", "image/bmp"),
        ".webp": ("WEBP", "image/webp"),
    }
    expected = allowed.get(suffix)
    if expected is None:
        raise ValueError("image must be JPEG, PNG, BMP, or WebP")
    expected_format, expected_mime = expected
    if content_type != expected_mime:
        raise ValueError("image MIME type does not match its filename")
    if not content:
        raise ValueError("image file is empty")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format != expected_format:
                    raise ValueError("image bytes do not match its filename")
                width, height = source.size
                if width * height > 25_000_000:
                    raise ValueError("image dimensions exceed the 25 megapixel limit")
                source.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("image dimensions are unsafe") from error
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise ValueError("image bytes are invalid or corrupted") from error


def _checkpoint_info(
    path: Path,
    root_dir: Path,
    deployed_path: Path | None,
    deployed_sha256: str | None,
    expected_class_order: tuple[str, ...],
) -> CheckpointPublic:
    import torch

    resolved = path.resolve()
    rel_path = resolved.relative_to(root_dir).as_posix()
    is_deployed = deployed_path is not None and resolved == deployed_path
    model_name = None
    model_version = None
    experiment_type = None
    created_at = None
    eligible = False
    validation_error = None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            model_name = payload.get("model_name")
            model_version = payload.get("model_version")
            created_at = payload.get("created_at")
            metadata = payload.get("metadata", {})
            if isinstance(metadata, dict):
                experiment_type = metadata.get("experiment_type")
            eligible = (
                payload.get("checkpoint_kind") == "cropfed_image_classifier"
                and isinstance(payload.get("state_dict"), dict)
                and payload.get("class_order") == list(expected_class_order)
            )
            if not eligible:
                validation_error = "unsupported checkpoint envelope"
        else:
            validation_error = "checkpoint payload is not an object"
    except Exception as error:
        validation_error = type(error).__name__
    return CheckpointPublic(
        filename=path.name,
        path=rel_path,
        size_bytes=path.stat().st_size,
        model_name=model_name,
        model_version=model_version,
        experiment_type=experiment_type,
        created_at=created_at,
        sha256=deployed_sha256 if is_deployed else None,
        deployed=is_deployed,
        eligible_for_deployment=eligible,
        validation_error=validation_error,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
