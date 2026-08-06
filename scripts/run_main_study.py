"""Run the full main-study matrix across FL scenarios.

Each scenario runs Flower Simulation against the real PlantVillage profiles.
Flower reads run_config from [tool.flwr.app.config] in pyproject.toml, so
this script rewrites that section per scenario, invokes Flower, and restores
the original section afterwards.

Usage:
    python scripts/run_main_study.py \\
        --output-root artifacts/main-study-seed2026 \\
        --rounds 10 --local-epochs 1 --num-cpus 4 \\
        [--only FL-IID-AVG]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from flwr.simulation import run_simulation  # noqa: E402

from cropfed.constants import DatasetTaxonomy, taxonomy_from_scope  # noqa: E402
from cropfed.experiments.centralized import run_centralized  # noqa: E402
from cropfed.experiments.local_only import run_local_only  # noqa: E402
from cropfed.flower.client_app import app as client_app  # noqa: E402
from cropfed.flower.server_app import app as server_app  # noqa: E402
from cropfed.flower.smoke import validate_run_artifacts  # noqa: E402

NUM_CLIENTS = 4
DEFAULT_PROXIMAL_MU = 0.01
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
RUN_CONFIG_HEADER = "[tool.flwr.app.config]"


SCENARIOS: list[dict[str, object]] = [
    {"id": "CEN-MBV3", "mode": "centralized", "profile": "iid", "algorithm": "centralized"},
    {"id": "LOC-MBV3", "mode": "local-only", "profile": "alpha-0.5", "algorithm": "local-only"},
    {"id": "FL-IID-AVG", "mode": "federated", "profile": "iid", "algorithm": "fedavg"},
    {"id": "FL-A100-AVG", "mode": "federated", "profile": "alpha-100", "algorithm": "fedavg"},
    {"id": "FL-A05-AVG", "mode": "federated", "profile": "alpha-0.5", "algorithm": "fedavg"},
    {"id": "FL-A01-AVG", "mode": "federated", "profile": "alpha-0.1", "algorithm": "fedavg"},
    {"id": "FL-A05-PROX", "mode": "federated", "profile": "alpha-0.5", "algorithm": "fedprox"},
    {"id": "FL-A01-PROX", "mode": "federated", "profile": "alpha-0.1", "algorithm": "fedprox"},
]


def _resolve_profile_dir(profile: str, profiles_root: Path) -> Path:
    base = profiles_root
    if profile == "iid":
        return base / "iid"
    if profile.startswith("alpha-"):
        return base / f"dirichlet-alpha-{profile.removeprefix('alpha-')}"
    raise ValueError(f"unknown profile: {profile}")


def _profile_alpha(profile_dir: Path) -> float | None:
    prefix = "dirichlet-alpha-"
    if profile_dir.name.startswith(prefix):
        return float(profile_dir.name.removeprefix(prefix))
    return None


def _scenario_tag(scenario_id: str, *, seed: int) -> str:
    return f"{scenario_id.lower().replace('-', '_')}_seed{seed}"


def _read_pyproject() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _write_pyproject(contents: str) -> None:
    PYPROJECT.write_text(contents, encoding="utf-8")


def _replace_run_config(
    pyproject_text: str, new_entries: dict[str, object]
) -> str:
    """Replace the [tool.flwr.app.config] block with new key=value lines."""

    pattern = re.compile(
        r"(\[tool\.flwr\.app\.config\]\r?\n)(?:.*\r?\n)*?(?=\r?\n\[|\Z)", re.MULTILINE
    )
    body_lines = [f"{key} = {_format_value(new_entries[key])}" for key in new_entries]
    new_block = RUN_CONFIG_HEADER + "\n" + "\n".join(body_lines) + "\n"
    return pattern.sub(new_block, pyproject_text)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


class PyprojectConfig:
    """Context manager that swaps [tool.flwr.app.config] in pyproject.toml."""

    def __init__(self, entries: dict[str, object]) -> None:
        self.entries = entries
        self._original: str | None = None

    def __enter__(self) -> None:
        self._original = _read_pyproject()
        _write_pyproject(_replace_run_config(self._original, self.entries))

    def __exit__(self, *_exc: object) -> None:
        assert self._original is not None
        _write_pyproject(self._original)


def _run_centralized(
    *,
    profile_dir: Path,
    output_dir: Path,
    rounds: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    model_name: str,
    taxonomy: DatasetTaxonomy,
    research_result_valid: bool,
    protocol_lock: Path | None,
) -> dict[str, object]:
    pooled_manifest = profile_dir / "pooled_train_manifest.csv"
    validation_manifest = profile_dir / "validation_manifest.csv"
    test_manifest = profile_dir / "test_manifest.csv"
    required = (pooled_manifest, validation_manifest, test_manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing centralized manifests: {missing}")
    return run_centralized(
        train_manifest=pooled_manifest,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        model_name=model_name,
        epochs=rounds,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained=True,
        seed=seed,
        output_dir=output_dir,
        research_result_valid=research_result_valid,
        protocol_lock=protocol_lock,
        class_names=taxonomy.class_names,
        class_groups=taxonomy.class_groups,
    )


def _run_local_only(
    *,
    profile_dir: Path,
    output_dir: Path,
    rounds: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    model_name: str,
    taxonomy: DatasetTaxonomy,
    research_result_valid: bool,
    protocol_lock: Path | None,
) -> dict[str, object]:
    return run_local_only(
        client_data_root=profile_dir / "clients",
        test_manifest=profile_dir / "test_manifest.csv",
        num_clients=NUM_CLIENTS,
        model_name=model_name,
        epochs=rounds,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained=True,
        seed=seed,
        output_dir=output_dir,
        partition_kind=(
            "dirichlet" if profile_dir.name != "iid" else "iid"
        ),
        dirichlet_alpha=_profile_alpha(profile_dir),
        research_result_valid=research_result_valid,
        protocol_lock=protocol_lock,
        class_names=taxonomy.class_names,
        class_groups=taxonomy.class_groups,
    )


def _run_federated(
    *,
    profile_dir: Path,
    output_dir: Path,
    algorithm: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    proximal_mu: float,
    seed: int,
    pretrained: bool,
    num_cpus: int,
    num_gpus: float,
    model_name: str,
    taxonomy: DatasetTaxonomy,
    research_result_valid: bool,
    protocol_lock: Path | None,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_kind = (
        "dirichlet" if profile_dir.name != "iid" else "iid"
    )
    dirichlet_alpha = _profile_alpha(profile_dir)

    run_config_entries = {
        "num-server-rounds": rounds,
        "num-clients": NUM_CLIENTS,
        "fraction-train": 1.0,
        "fraction-evaluate": 1.0,
        "local-epochs": local_epochs,
        "batch-size": batch_size,
        "learning-rate": learning_rate,
        "seed": seed,
        "algorithm": algorithm,
        "proximal-mu": proximal_mu,
        "partition-kind": partition_kind,
        "dirichlet-alpha": dirichlet_alpha or 0.0,
        "model-name": model_name,
        "taxonomy-scope": taxonomy.scope,
        "pretrained": pretrained,
        "client-data-root": (profile_dir / "clients").as_posix(),
        "global-test-manifest": (profile_dir / "test_manifest.csv").as_posix(),
        "save-model": True,
        "output-dir": output_dir.as_posix(),
        "result-kind": "federated_image_main_study",
        "research-result-valid": research_result_valid,
        "protocol-lock": protocol_lock.as_posix() if protocol_lock else "",
    }

    started = time.perf_counter()
    with PyprojectConfig(run_config_entries):
        print(f"   run_config keys: {list(run_config_entries.keys())}")
        # Verify pyproject on disk has seed
        on_disk = _read_pyproject()
        if "seed" not in on_disk:
            print("   !!! seed NOT in on-disk pyproject after patch")
            print(on_disk[on_disk.index("[tool.flwr.app.config]"):][:500])
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=NUM_CLIENTS,
            backend_name="ray",
            backend_config={
                "client_resources": {
                    "num_cpus": 1,
                    "num_gpus": min(1.0, num_gpus) if num_gpus > 0 else 0.0,
                },
                "init_args": {
                    "num_cpus": num_cpus,
                    "num_gpus": num_gpus,
                    "log_to_driver": True,
                    "include_dashboard": False,
                },
            },
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
        proximal_mu=proximal_mu,
        log_text=log_text,
        expected_class_order=taxonomy.class_names,
    )
    validation["elapsed_seconds"] = elapsed
    return validation


def run_scenario(
    scenario: dict[str, object],
    *,
    output_root: Path,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    pretrained: bool,
    num_cpus: int,
    num_gpus: float,
    profiles_root: Path,
    model_name: str,
    taxonomy: DatasetTaxonomy,
    research_result_valid: bool,
    protocol_lock_root: Path | None,
    proximal_mu: float,
) -> dict[str, object]:
    profile_dir = _resolve_profile_dir(str(scenario["profile"]), profiles_root)
    output_dir = output_root / _scenario_tag(str(scenario["id"]), seed=seed)
    protocol_lock = (
        protocol_lock_root / f"{str(scenario['id']).lower()}.json"
        if protocol_lock_root is not None
        else None
    )
    print(f"\n== {scenario['id']} ({scenario['mode']}) -> {output_dir.name} ==")

    if scenario["mode"] == "centralized":
        result = _run_centralized(
            profile_dir=profile_dir,
            output_dir=output_dir,
            rounds=rounds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            model_name=model_name,
            taxonomy=taxonomy,
            research_result_valid=research_result_valid,
            protocol_lock=protocol_lock,
        )
    elif scenario["mode"] == "local-only":
        result = _run_local_only(
            profile_dir=profile_dir,
            output_dir=output_dir,
            rounds=rounds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            model_name=model_name,
            taxonomy=taxonomy,
            research_result_valid=research_result_valid,
            protocol_lock=protocol_lock,
        )
    elif scenario["mode"] == "federated":
        result = _run_federated(
            profile_dir=profile_dir,
            output_dir=output_dir,
            algorithm=str(scenario["algorithm"]),
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            proximal_mu=proximal_mu,
            seed=seed,
            pretrained=pretrained,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            model_name=model_name,
            taxonomy=taxonomy,
            research_result_valid=research_result_valid,
            protocol_lock=protocol_lock,
        )
    else:
        raise ValueError(f"unknown mode: {scenario['mode']}")

    print(
        f"   elapsed={result.get('elapsed_seconds', 0):.1f}s "
        f"status={result.get('status', 'unknown')}"
    )
    return {
        "scenario_id": scenario["id"],
        "mode": scenario["mode"],
        "algorithm": scenario["algorithm"],
        "profile": scenario["profile"],
        "output_dir": str(output_dir),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "flower-profiles-full",
    )
    parser.add_argument(
        "--taxonomy",
        choices=["plantvillage-full", "tomato"],
        default="plantvillage-full",
    )
    parser.add_argument(
        "--model",
        choices=["mobilenet_v3_small", "efficientnet_lite0", "mobilenet_v2"],
        default="mobilenet_v3_small",
    )
    parser.add_argument("--proximal-mu", type=float, default=DEFAULT_PROXIMAL_MU)
    parser.add_argument("--research-run", action="store_true")
    parser.add_argument(
        "--protocol-lock-root",
        type=Path,
        help="directory containing one locked JSON file named <scenario-id>.json",
    )
    parser.add_argument("--only", choices=[s["id"] for s in SCENARIOS])
    args = parser.parse_args()

    if args.research_run and args.protocol_lock_root is None:
        parser.error("--research-run requires --protocol-lock-root")
    if args.num_gpus < 0:
        parser.error("--num-gpus cannot be negative")

    taxonomy = taxonomy_from_scope(args.taxonomy)
    profiles_root = args.profiles_root.expanduser().resolve()
    protocol_lock_root = (
        args.protocol_lock_root.expanduser().resolve()
        if args.protocol_lock_root is not None
        else None
    )

    output_root: Path = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        print(f"refusing to overwrite non-empty output root: {output_root}")
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    selected = SCENARIOS
    if args.only:
        selected = [s for s in SCENARIOS if s["id"] == args.only]

    results: list[dict[str, object]] = []
    started = time.perf_counter()
    for scenario in selected:
        try:
            row = run_scenario(
                scenario,
                output_root=output_root,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                pretrained=True,
                num_cpus=args.num_cpus,
                num_gpus=args.num_gpus,
                profiles_root=profiles_root,
                model_name=args.model,
                taxonomy=taxonomy,
                research_result_valid=args.research_run,
                protocol_lock_root=protocol_lock_root,
                proximal_mu=args.proximal_mu,
            )
        except Exception as error:
            row = {
                "scenario_id": scenario["id"],
                "mode": scenario["mode"],
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"   FAILED: {row['error']}")
        results.append(row)

    total_elapsed = time.perf_counter() - started
    summary_path = output_root / "main_study_summary.json"
    all_succeeded = all(row.get("status") != "failed" for row in results)
    summary = {
        "status": "completed" if all_succeeded else "failed",
        "result_kind": (
            "federated_image_research_candidate"
            if args.research_run and all_succeeded
            else "federated_image_main_study_pilot"
        ),
        "research_result_valid": bool(args.research_run and all_succeeded),
        "num_clients": NUM_CLIENTS,
        "taxonomy_scope": taxonomy.scope,
        "class_order": list(taxonomy.class_names),
        "config": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "pretrained": True,
            "proximal_mu": args.proximal_mu,
            "model": args.model,
            "profiles_root": str(profiles_root),
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
        },
        "total_elapsed_seconds": total_elapsed,
        "scenarios": results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nMain study summary written to {summary_path}")
    return 0 if all_succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
