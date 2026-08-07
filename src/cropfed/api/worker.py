"""Database-backed Flower worker kept outside the FastAPI web process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from cropfed.api.db import engine
from cropfed.api.migrate import upgrade_database
from cropfed.api.models import ExperimentRecord
from cropfed.api.results import replace_client_history, replace_round_history
from cropfed.api.settings import Settings, settings
from cropfed.constants import taxonomy_from_scope
from cropfed.data.audit import audit_prepared_data, write_audit_report
from cropfed.flower.smoke import validate_run_artifacts


@dataclass(frozen=True, slots=True)
class FlowerRunSpec:
    experiment_id: str
    algorithm: str
    partition_kind: str
    num_clients: int
    num_rounds: int
    local_epochs: int
    learning_rate: float
    batch_size: int
    dirichlet_alpha: float
    proximal_mu: float
    seed: int


FlowerExecutor = Callable[[FlowerRunSpec, Settings], dict[str, Any]]


def data_profile_name(partition_kind: str, dirichlet_alpha: float) -> str:
    """Map the whitelisted HTTP config to a server-owned dataset profile."""

    if partition_kind == "iid":
        return "iid"
    if partition_kind == "quantity_skew":
        return "quantity-skew"
    if partition_kind == "feature_skew":
        return "feature-skew"
    if partition_kind != "dirichlet":
        raise ValueError(
            "partition_kind must be 'iid', 'dirichlet', 'quantity_skew', "
            "or 'feature_skew'"
        )
    if isclose(dirichlet_alpha, 0.5, rel_tol=0.0, abs_tol=1e-9):
        return "dirichlet-alpha-0.5"
    if isclose(dirichlet_alpha, 0.1, rel_tol=0.0, abs_tol=1e-9):
        return "dirichlet-alpha-0.1"
    raise ValueError("Flower MVP supports Dirichlet alpha 0.5 or 0.1")


def claim_next_flower_experiment(database_engine: Engine) -> FlowerRunSpec | None:
    """Atomically claim the oldest queued Flower experiment."""

    with Session(database_engine) as session:
        candidate = session.exec(
            select(ExperimentRecord)
            .where(
                ExperimentRecord.status == "queued",
                ExperimentRecord.execution_mode == "flower",
            )
            .order_by(ExperimentRecord.created_at.asc())
            .limit(1)
        ).first()
        if candidate is None:
            return None

        claimed_at = datetime.now(UTC)
        result = session.exec(
            update(ExperimentRecord)
            .where(
                ExperimentRecord.id == candidate.id,
                ExperimentRecord.status == "queued",
            )
            .values(status="running", updated_at=claimed_at)
        )
        session.commit()
        if result.rowcount != 1:
            return None
        session.refresh(candidate)
        return _to_run_spec(candidate)


def run_worker_once(
    *,
    database_engine: Engine = engine,
    application_settings: Settings = settings,
    executor: FlowerExecutor | None = None,
) -> bool:
    """Claim and execute at most one job; return whether a job was claimed."""

    if not application_settings.flower_worker_enabled:
        raise RuntimeError("Flower worker is disabled by server configuration")
    spec = claim_next_flower_experiment(database_engine)
    if spec is None:
        return False
    selected_executor = executor or execute_flower_experiment
    try:
        result = selected_executor(spec, application_settings)
    except Exception as error:
        _finish_experiment(
            database_engine,
            spec.experiment_id,
            status_value="failed",
            error_message=f"{type(error).__name__}: {error}"[:2_000],
        )
    else:
        _finish_experiment(
            database_engine,
            spec.experiment_id,
            status_value="completed",
            result=result,
        )
    return True


def execute_flower_experiment(
    spec: FlowerRunSpec,
    application_settings: Settings,
) -> dict[str, Any]:
    """Audit the selected profile, launch Flower without a shell, and verify output."""

    if spec.num_clients != 4:
        raise ValueError("Flower MVP worker requires exactly four clients")
    project_root = application_settings.flower_project_dir.resolve()
    taxonomy = taxonomy_from_scope(application_settings.taxonomy_scope)
    profile_name = data_profile_name(spec.partition_kind, spec.dirichlet_alpha)
    data_root = _resolve_from(project_root, application_settings.flower_data_root)
    profile_root = data_root / profile_name
    client_data_root = profile_root / "clients"
    manifest_root = (
        profile_root
        if (profile_root / "train_manifest.csv").is_file()
        else profile_root / "processed"
    )
    train_manifest = manifest_root / "train_manifest.csv"
    test_manifest = manifest_root / "test_manifest.csv"

    output_root = _resolve_from(project_root, application_settings.flower_output_root)
    output_dir = _new_output_directory(output_root, spec.experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_report = audit_prepared_data(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        client_data_root=client_data_root,
        num_clients=spec.num_clients,
        class_names=taxonomy.class_names,
        dataset_root=_resolve_from(
            project_root, application_settings.flower_dataset_root
        ),
    )
    audit_path = output_dir / "pre_run_data_audit.json"
    write_audit_report(audit_report, audit_path)
    if audit_report["status"] != "passed":
        raise RuntimeError(f"pre-run data audit failed; see {audit_path.name}")

    log_path = output_dir / "flower.log"
    command = build_flower_command(
        spec,
        application_settings,
        project_root=project_root,
        client_data_root=client_data_root,
        test_manifest=test_manifest,
        output_dir=output_dir,
    )
    environment = _flower_environment(application_settings, project_root)
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=application_settings.flower_timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Flower exited with {completed.returncode}; see {log_path.name}"
        )

    validation = validate_run_artifacts(
        output_dir,
        algorithm=spec.algorithm,
        expected_clients=spec.num_clients,
        proximal_mu=spec.proximal_mu,
        log_text=log_path.read_text(encoding="utf-8"),
        expected_class_order=taxonomy.class_names,
    )
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Flower metrics.json was not created")
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics_payload.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("Flower metrics history is missing or empty")
    client_history = metrics_payload.get("client_history")
    _validate_client_history(
        client_history,
        num_clients=spec.num_clients,
        num_rounds=spec.num_rounds,
    )
    selection = metrics_payload.get("selection")
    if not isinstance(selection, dict) or not isinstance(
        selection.get("best_round"), int
    ):
        raise RuntimeError("Flower validation checkpoint selection is missing")
    global_test = metrics_payload.get("global_test")
    if not isinstance(global_test, dict) or "global_test_macro_f1" not in global_test:
        raise RuntimeError("Flower final global-test evaluation is missing")
    return {
        "result_kind": "flower_image_training_run",
        "research_result_valid": False,
        "research_validation_status": "pending_protocol_review",
        "data_profile": profile_name,
        "data_audit": {
            "status": audit_report["status"],
            "sha256": _sha256_file(audit_path),
            "num_train": audit_report["manifests"]["master_train"]["num_records"],
            "num_test": audit_report["manifests"]["global_test"]["num_records"],
            "class_order_matches": (
                tuple(audit_report["taxonomy"]["class_order"])
                == taxonomy.class_names
            ),
        },
        "flower": validation,
        "history": history,
        "client_history": client_history,
        "selection": selection,
        "global_test": global_test,
        "communication": metrics_payload.get("communication", {}),
        "artifact_directory": output_dir.name,
    }


def build_flower_command(
    spec: FlowerRunSpec,
    application_settings: Settings,
    *,
    project_root: Path,
    client_data_root: Path,
    test_manifest: Path,
    output_dir: Path,
    flower_executable: Path | None = None,
) -> list[str]:
    """Build a fixed argv list; no HTTP value can become shell syntax or a path."""

    executable = flower_executable or _flower_executable()
    federation_config = " ".join(
        [
            "num-supernodes=4",
            "verbose=true",
            "backend='ray'",
            "client-resources-num-cpus=1",
            "client-resources-num-gpus="
            f"{min(1.0, application_settings.flower_num_gpus)}",
            f"init-args-num-cpus={application_settings.flower_num_cpus}",
            f"init-args-num-gpus={application_settings.flower_num_gpus}",
            "init-args-log-to-driver=true",
        ]
    )
    dataset_root = _resolve_from(project_root, application_settings.flower_dataset_root)
    run_config = " ".join(
        [
            f"algorithm='{spec.algorithm}'",
            f"proximal-mu={spec.proximal_mu}",
            "scaffold-server-lr=1.0",
            "moon-temperature=0.5",
            "moon-mu=1.0",
            f"partition-kind='{spec.partition_kind}'",
            f"dirichlet-alpha={spec.dirichlet_alpha}",
            "num-clients=4",
            f"num-server-rounds={spec.num_rounds}",
            f"local-epochs={spec.local_epochs}",
            f"batch-size={spec.batch_size}",
            f"learning-rate={spec.learning_rate}",
            f"seed={spec.seed}",
            f"pretrained={str(application_settings.flower_pretrained).lower()}",
            f"model-name='{application_settings.flower_model_name}'",
            f"taxonomy-scope='{application_settings.taxonomy_scope}'",
            f"client-data-root={_toml_string(client_data_root)}",
            f"global-test-manifest={_toml_string(test_manifest)}",
            f"dataset-root={_toml_string(dataset_root)}",
            f"num-workers={application_settings.flower_num_workers}",
            f"output-dir={_toml_string(output_dir)}",
            "save-model=true",
        ]
    )
    return [
        str(executable),
        "run",
        str(project_root),
        application_settings.flower_superlink,
        "--stream",
        "--federation-config",
        federation_config,
        "--run-config",
        run_config,
    ]


def _flower_environment(
    application_settings: Settings,
    project_root: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(project_root / "src"), environment.get("PYTHONPATH", "")])
    )
    environment["VIRTUAL_ENV"] = sys.prefix
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", "")]
    )
    environment["FLWR_HOME"] = str(
        _resolve_from(project_root, application_settings.flower_home)
    )
    environment["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["RAY_DEDUP_LOGS"] = "0"
    return environment


def _flower_executable() -> Path:
    executable_name = "flwr.exe" if os.name == "nt" else "flwr"
    sibling = Path(sys.executable).with_name(executable_name)
    if sibling.is_file():
        return sibling
    discovered = shutil.which("flwr")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("flwr executable not found in the worker environment")


def _to_run_spec(record: ExperimentRecord) -> FlowerRunSpec:
    return FlowerRunSpec(
        experiment_id=record.id,
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


def _finish_experiment(
    database_engine: Engine,
    experiment_id: str,
    *,
    status_value: str,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    with Session(database_engine) as session:
        record = session.get(ExperimentRecord, experiment_id)
        if record is None:
            return
        record.status = status_value
        record.result_json = (
            json.dumps(result, ensure_ascii=False) if result is not None else None
        )
        if result is not None:
            replace_round_history(session, experiment_id, result)
            replace_client_history(session, experiment_id, result)
        record.error_message = error_message
        record.updated_at = datetime.now(UTC)
        session.add(record)
        session.commit()


def _validate_client_history(
    history: object,
    *,
    num_clients: int,
    num_rounds: int,
) -> None:
    if not isinstance(history, list):
        raise RuntimeError("Flower client history is missing")
    expected = num_clients * num_rounds * 2
    if len(history) != expected:
        raise RuntimeError(
            f"Flower client history has {len(history)} entries; expected {expected}"
        )
    identities: set[tuple[int, int, str]] = set()
    for item in history:
        if not isinstance(item, dict):
            raise RuntimeError("Flower client history entry is not an object")
        identity = (int(item["round"]), int(item["client_id"]), str(item["phase"]))
        identities.add(identity)
    expected_identities = {
        (round_number, client_id, phase)
        for round_number in range(1, num_rounds + 1)
        for client_id in range(num_clients)
        for phase in ("train", "evaluate")
    }
    if identities != expected_identities:
        raise RuntimeError("Flower client history identities are incomplete")


def _resolve_from(project_root: Path, configured: Path) -> Path:
    if configured.is_absolute():
        return configured.resolve()
    return (project_root / configured).resolve()


def _new_output_directory(output_root: Path, experiment_id: str) -> Path:
    primary = output_root / experiment_id
    if not primary.exists() or not any(primary.iterdir()):
        return primary
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    retry = output_root / f"{experiment_id}-retry-{suffix}"
    if retry.exists():
        raise FileExistsError(f"Flower retry output already exists: {retry}")
    return retry


def _toml_string(path: Path) -> str:
    value = path.resolve().as_posix()
    if "'" in value:
        raise ValueError("Flower paths cannot contain a single quote")
    return f"'{value}'"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup_stale_running_experiments(
    database_engine: Engine = engine,
    max_stale_seconds: float = 3600.0,
) -> int:
    """Mark experiments left in 'running' status for over max_stale_seconds as failed."""

    cutoff = datetime.now(UTC)
    count = 0
    with Session(database_engine) as session:
        records = session.exec(
            select(ExperimentRecord).where(ExperimentRecord.status == "running")
        ).all()
        for record in records:
            updated_at = (
                record.updated_at.replace(tzinfo=UTC)
                if record.updated_at.tzinfo is None
                else record.updated_at
            )
            elapsed = (cutoff - updated_at).total_seconds()
            if elapsed > max_stale_seconds:
                record.status = "failed"
                record.error_message = (
                    f"StaleWorkerError: experiment remained in running status "
                    f"for {elapsed:.0f}s without worker heartbeat"
                )
                record.updated_at = cutoff
                session.add(record)
                count += 1
        if count > 0:
            session.commit()
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    upgrade_database(database_url=settings.database_url)
    if not settings.flower_worker_enabled:
        print("Flower worker is disabled; set CROPFED_FLOWER_WORKER_ENABLED=true")
        return 2
    cleanup_stale_running_experiments(engine)
    poll_interval = args.poll_interval or settings.flower_poll_interval
    if poll_interval <= 0:
        raise ValueError("poll interval must be positive")
    while True:
        claimed = run_worker_once()
        if args.once:
            return 0
        if not claimed:
            time.sleep(poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
