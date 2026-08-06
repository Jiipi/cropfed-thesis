"""Run and verify Flower FedAvg/FedProx with four synthetic image clients."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# The repository source is intentionally bootstrapped before project imports.
from create_flower_smoke_fixture import create_fixture  # noqa: E402

from cropfed.constants import TOMATO_CLASSES  # noqa: E402
from cropfed.flower.smoke import (  # noqa: E402
    compare_checkpoint_states,
    strip_ansi,
    validate_run_artifacts,
)

NUM_CLIENTS = 4
PROXIMAL_MU = 0.01


def _flower_executable() -> Path:
    executable_name = "flwr.exe" if os.name == "nt" else "flwr"
    sibling = Path(sys.executable).with_name(executable_name)
    if sibling.is_file():
        return sibling
    discovered = shutil.which("flwr")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError(
        "flwr executable not found; run this script from the Flower environment"
    )


def _run_config(
    *,
    algorithm: str,
    fixture_root: Path,
    output_dir: Path,
) -> str:
    paths = (fixture_root / "clients", fixture_root / "processed" / "test_manifest.csv", output_dir)
    if any("'" in path.as_posix() for path in paths):
        raise ValueError("Flower smoke paths cannot contain a single quote")
    return " ".join(
        [
            f"algorithm='{algorithm}'",
            f"proximal-mu={PROXIMAL_MU}",
            "partition-kind='dirichlet'",
            "dirichlet-alpha=0.5",
            "num-clients=4",
            "num-server-rounds=1",
            "local-epochs=1",
            "batch-size=2",
            "learning-rate=0.001",
            "seed=2026",
            "pretrained=false",
            f"client-data-root='{(fixture_root / 'clients').as_posix()}'",
            "global-test-manifest="
            f"'{(fixture_root / 'processed' / 'test_manifest.csv').as_posix()}'",
            f"output-dir='{output_dir.as_posix()}'",
            "save-model=true",
        ]
    )


def _federation_config() -> str:
    return " ".join(
        [
            "num-supernodes=4",
            "verbose=true",
            "backend='ray'",
            "client-resources-num-cpus=1",
            "client-resources-num-gpus=0.0",
            "init-args-num-cpus=2",
            "init-args-num-gpus=0",
            "init-args-log-to-driver=true",
        ]
    )


def run_one(
    *,
    project_root: Path,
    fixture_root: Path,
    algorithm: str,
    flwr_home: Path,
    verify_existing: bool = False,
) -> dict[str, object]:
    output_dir = fixture_root / "runs" / algorithm
    log_path = output_dir / "flower.log"
    if verify_existing:
        if not log_path.is_file():
            raise FileNotFoundError(f"Flower smoke log not found: {log_path}")
        validation = validate_run_artifacts(
            output_dir,
            algorithm=algorithm,
            expected_clients=NUM_CLIENTS,
            proximal_mu=PROXIMAL_MU,
            expected_class_order=TOMATO_CLASSES,
            log_text=log_path.read_text(encoding="utf-8"),
        )
        validation["log"] = log_path.name
        return validation
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite an existing Flower run: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(_flower_executable()),
        "run",
        ".",
        "local",
        "--stream",
        "--federation-config",
        _federation_config(),
        "--run-config",
        _run_config(
            algorithm=algorithm,
            fixture_root=fixture_root,
            output_dir=output_dir,
        ),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(project_root / "src"), environment.get("PYTHONPATH", "")])
    )
    environment["VIRTUAL_ENV"] = sys.prefix
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", "")]
    )
    environment["FLWR_HOME"] = str(flwr_home)
    environment["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["RAY_DEDUP_LOGS"] = "0"

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    clean_log = strip_ansi(completed.stdout + completed.stderr)
    log_path.write_text(clean_log, encoding="utf-8")
    _safe_console_write(clean_log)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Flower {algorithm} exited with {completed.returncode}; see {log_path}"
        )

    validation = validate_run_artifacts(
        output_dir,
        algorithm=algorithm,
        expected_clients=NUM_CLIENTS,
        proximal_mu=PROXIMAL_MU,
        expected_class_order=TOMATO_CLASSES,
        log_text=clean_log,
    )
    validation["elapsed_seconds"] = elapsed
    validation["log"] = log_path.name
    return validation


def _safe_console_write(value: str) -> None:
    """Write logs even when a Windows console uses a legacy code page."""

    encoding = sys.stdout.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding)
    print(safe_value, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument(
        "--algorithm",
        choices=["fedavg", "fedprox", "both"],
        default="both",
    )
    parser.add_argument("--flwr-home", type=Path, default=Path(".flwr"))
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="validate existing logs/artifacts without starting Flower",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    fixture_root = args.fixture_root.resolve()
    fixture_path = fixture_root / "fixture.json"
    if not fixture_path.is_file():
        create_fixture(fixture_root, num_clients=NUM_CLIENTS)

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("fixture_kind") != "synthetic_images_for_integration_only":
        raise RuntimeError("refusing to treat a non-smoke dataset as a smoke fixture")
    algorithms = ["fedavg", "fedprox"] if args.algorithm == "both" else [args.algorithm]
    results = [
        run_one(
            project_root=PROJECT_ROOT,
            fixture_root=fixture_root,
            algorithm=algorithm,
            flwr_home=args.flwr_home.resolve(),
            verify_existing=args.verify_existing,
        )
        for algorithm in algorithms
    ]
    summary: dict[str, object] = {
        "status": "passed",
        "result_kind": "synthetic_images_integration_only",
        "research_result_valid": False,
        "num_clients": NUM_CLIENTS,
        "algorithms": results,
    }
    if len(algorithms) == 2:
        comparison = compare_checkpoint_states(
            fixture_root / "runs" / "fedavg" / "global_model.pt",
            fixture_root / "runs" / "fedprox" / "global_model.pt",
        )
        if comparison["different_tensors"] == 0:
            raise RuntimeError(
                "FedAvg and FedProx weights are identical; the smoke fixture did not "
                "exercise a non-zero proximal update"
            )
        summary["algorithm_state_comparison"] = comparison
    summary_path = fixture_root / "flower_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Flower smoke passed; summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
