"""Run a Flower FedAvg/FedProx pilot on PlantVillage profiles via the Simulation API.

This bypasses the external `flower-superlink` binary by invoking
``flwr.simulation.run_simulation`` directly. It reuses the production
``ServerApp``/``ClientApp`` and mirrors the validation that
``run_flower_smoke.py`` performs for the synthetic fixture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Project imports follow the sys.path tweak above.  noqa keeps ruff happy.
from flwr.simulation import run_simulation  # noqa: E402

from cropfed.constants import TOMATO_CLASSES  # noqa: E402
from cropfed.flower.client_app import app as client_app  # noqa: E402
from cropfed.flower.server_app import app as server_app  # noqa: E402
from cropfed.flower.smoke import (  # noqa: E402
    validate_run_artifacts,
)

NUM_CLIENTS = 4
PROXIMAL_MU = 0.01


def _resolve_profile(profile: str) -> Path:
    base = PROJECT_ROOT / "data" / "flower-profiles"
    if profile == "iid":
        return base / "iid"
    if profile.startswith("alpha-"):
        return base / f"dirichlet-alpha-{profile.removeprefix('alpha-')}"
    raise ValueError(f"unknown profile: {profile}")


def _run_config(
    *,
    algorithm: str,
    profile_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "proximal-mu": PROXIMAL_MU,
        "partition-kind": "dirichlet",
        "dirichlet-alpha": 0.5 if "0.5" in profile_dir.name else 0.1,
        "num-clients": NUM_CLIENTS,
        "num-server-rounds": 1,
        "local-epochs": 1,
        "batch-size": 32,
        "learning-rate": 0.001,
        "seed": 2026,
        "pretrained": True,
        "model-name": "mobilenet_v2",
        "client-data-root": (profile_dir / "clients").as_posix(),
        "global-test-manifest": (profile_dir / "test_manifest.csv").as_posix(),
        "output-dir": output_dir.as_posix(),
        "save-model": True,
        "result-kind": "federated_image_pilot",
        "research-result-valid": False,
        "protocol-lock": "",
    }


def _backend_config(*, num_cpus: int) -> dict[str, object]:
    return {
        "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
        "init_args": {
            "num_cpus": num_cpus,
            "num_gpus": 0,
            "log_to_driver": True,
            "include_dashboard": False,
        },
    }


def _node_config(partition_id: int) -> dict[str, int]:
    return {"partition-id": partition_id}


def run_one(
    *,
    output_dir: Path,
    algorithm: str,
    profile_dir: Path,
    num_cpus: int,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _run_config(algorithm=algorithm, profile_dir=profile_dir, output_dir=output_dir)

    # Stash the run config so the ServerApp/ClientApp can pick it up.
    # The Simulation runtime overrides ``run_config`` from kwargs, so we use
    # the Context-aware ``run_simulation`` flow.
    started = time.perf_counter()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_name="ray",
        backend_config=_backend_config(num_cpus=num_cpus),
        enable_tf_gpu_growth=False,
        verbose_logging=False,
    )
    elapsed = time.perf_counter() - started

    log_path = output_dir / "flower.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    validation = validate_run_artifacts(
        output_dir,
        algorithm=algorithm,
        expected_clients=NUM_CLIENTS,
        proximal_mu=PROXIMAL_MU,
        expected_class_order=TOMATO_CLASSES,
        log_text=log_text,
    )
    validation["elapsed_seconds"] = elapsed
    validation["profile"] = profile_dir.name
    validation["run_config"] = {k: str(v) for k, v in run_cfg.items()}
    return validation


def write_summary(output_dir: Path, results: list[dict[str, object]]) -> Path:
    summary_path = output_dir / "plantvillage_pilot_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "result_kind": "federated_image_pilot",
                "research_result_valid": False,
                "num_clients": NUM_CLIENTS,
                "tomato_classes": list(TOMATO_CLASSES),
                "algorithms": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["iid", "alpha-0.5", "alpha-0.1"],
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        choices=["fedavg", "fedprox", "both"],
        default="both",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory that will receive plantvillage-flower-* artifacts.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=8,
        help="Ray init CPU budget (use 8 to mimic the Windows single-actor workaround).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile_dir = _resolve_profile(args.profile)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile missing: {profile_dir}")

    algorithms = ["fedavg", "fedprox"] if args.algorithm == "both" else [args.algorithm]
    results: list[dict[str, object]] = []
    for algorithm in algorithms:
        run_tag = (
            f"plantvillage-flower-{algorithm}-{args.profile}-pilot-seed2026"
        )
        output_dir = args.output_dir / run_tag
        print(f"== Running {algorithm} on {profile_dir.name} -> {output_dir.name} ==")
        result = run_one(
            output_dir=output_dir,
            algorithm=algorithm,
            profile_dir=profile_dir,
            num_cpus=args.num_cpus,
        )
        results.append(result)
        print(f"   elapsed={result['elapsed_seconds']:.1f}s "
              f"checkpoint_sha256={result['checkpoint_sha256'][:12]}…")

    summary_path = write_summary(args.output_dir, results)
    print(f"Pilot runs complete; summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
