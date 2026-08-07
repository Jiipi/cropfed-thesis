"""Exercise API -> SQLite queue -> external worker -> Flower end to end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from create_flower_smoke_fixture import create_fixture  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import create_engine  # noqa: E402

from cropfed.api.main import create_app  # noqa: E402
from cropfed.api.settings import Settings  # noqa: E402
from cropfed.api.worker import run_worker_once  # noqa: E402


def run_smoke(
    output_root: Path,
    *,
    algorithm: str,
    flwr_home: Path,
) -> dict[str, object]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"smoke output must be new or empty: {output_root}")

    profile_root = output_root / "profiles" / "dirichlet-alpha-0.5"
    create_fixture(profile_root)
    database_path = output_root / "api-worker-smoke.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    database_engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
    application_settings = Settings(
        database_url=database_url,
        cors_origins=(),
        flower_worker_enabled=True,
        flower_project_dir=PROJECT_ROOT,
        flower_data_root=output_root / "profiles",
        # The fixture's manifests are relative to its own root, which is the
        # profile directory here; a real deployment points this at data/raw.
        flower_dataset_root=profile_root,
        flower_output_root=output_root / "runs",
        flower_home=flwr_home.resolve(),
        flower_superlink="local",
        flower_num_cpus=2,
        flower_timeout_seconds=600,
        flower_poll_interval=0.1,
        flower_pretrained=False,
    )
    try:
        with TestClient(
            create_app(
                database_engine=database_engine,
                application_settings=application_settings,
            )
        ) as client:
            created = client.post(
                "/api/v1/experiments",
                json={
                    "name": f"API worker {algorithm} integration smoke",
                    "execution_mode": "flower",
                    "algorithm": algorithm,
                    "partition_kind": "dirichlet",
                    "dirichlet_alpha": 0.5,
                    "num_clients": 4,
                    "num_rounds": 1,
                    "local_epochs": 1,
                    "learning_rate": 0.001,
                    "batch_size": 2,
                    "proximal_mu": 0.01,
                    "seed": 2026,
                },
            )
            if created.status_code != 201:
                raise RuntimeError(f"API create failed: {created.text}")
            experiment_id = created.json()["id"]
            started = client.post(f"/api/v1/experiments/{experiment_id}/start")
            if started.status_code != 202 or started.json()["status"] != "queued":
                raise RuntimeError(f"API queue failed: {started.text}")

            claimed = run_worker_once(
                database_engine=database_engine,
                application_settings=application_settings,
            )
            completed = client.get(f"/api/v1/experiments/{experiment_id}")
            if completed.status_code != 200:
                raise RuntimeError(f"API read failed: {completed.text}")
            record = completed.json()
            rounds_response = client.get(
                f"/api/v1/experiments/{experiment_id}/rounds"
            )
            if rounds_response.status_code != 200:
                raise RuntimeError(
                    f"API round-history read failed: {rounds_response.text}"
                )
            round_history = rounds_response.json()
            clients_response = client.get(
                f"/api/v1/experiments/{experiment_id}/clients"
            )
            if clients_response.status_code != 200:
                raise RuntimeError(
                    f"API client-history read failed: {clients_response.text}"
                )
            client_history = clients_response.json()
    finally:
        database_engine.dispose()

    if not claimed or record["status"] != "completed":
        raise RuntimeError(
            "API worker smoke failed: "
            f"claimed={claimed}, status={record['status']}, "
            f"error={record['error_message']}"
        )
    result = record["result"]
    if result.get("research_result_valid") is not False:
        raise RuntimeError("integration smoke result must never be research-valid")
    expected_rounds = record["num_rounds"] + 1  # Includes central evaluation round 0.
    if round_history.get("storage") != "database":
        raise RuntimeError("round history was not served from normalized database rows")
    if len(round_history.get("items", [])) != expected_rounds:
        raise RuntimeError("normalized round history has the wrong number of entries")
    summaries = round_history.get("summaries", [])
    if len(summaries) != expected_rounds:
        raise RuntimeError("normalized round summaries are missing")
    if [item.get("round") for item in summaries] != list(range(expected_rounds)):
        raise RuntimeError("normalized round summaries are not ordered from round 0")
    if any(item.get("macro_f1") is None for item in summaries):
        raise RuntimeError("normalized round summaries are missing central Macro-F1")
    expected_client_entries = record["num_clients"] * record["num_rounds"] * 2
    client_items = client_history.get("items", [])
    if len(client_items) != expected_client_entries:
        raise RuntimeError("normalized client metric rows are incomplete")
    client_identities = {
        (item["round"], item["client_id"], item["phase"])
        for item in client_items
    }
    expected_client_identities = {
        (round_number, client_id, phase)
        for round_number in range(1, record["num_rounds"] + 1)
        for client_id in range(record["num_clients"])
        for phase in ("train", "evaluate")
    }
    if client_identities != expected_client_identities:
        raise RuntimeError("normalized client metric identities are incomplete")
    communication = result.get("communication", {})
    if int(communication.get("payload_total_bytes", 0)) <= 0:
        raise RuntimeError("Flower communication bytes were not persisted")

    summary: dict[str, object] = {
        "status": "passed",
        "result_kind": "api_worker_flower_integration_only",
        "research_result_valid": False,
        "execution_mode": record["execution_mode"],
        "algorithm": record["algorithm"],
        "num_clients": record["num_clients"],
        "api_lifecycle": ["draft", "queued", "running", "completed"],
        "worker_claimed": claimed,
        "round_store": {
            "storage": round_history["storage"],
            "entries": len(round_history["items"]),
            "rounds": [item["round"] for item in summaries],
        },
        "client_metric_store": {
            "entries": len(client_items),
            "identities_complete": True,
        },
        "communication": communication,
        "data_audit": result["data_audit"],
        "flower": result["flower"],
    }
    summary_path = output_root / "api_worker_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**summary, "summary": str(summary_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=["fedavg", "fedprox"], default="fedavg"
    )
    parser.add_argument("--flwr-home", type=Path, default=Path(".flwr"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_smoke(
        args.output_root,
        algorithm=args.algorithm,
        flwr_home=args.flwr_home,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
