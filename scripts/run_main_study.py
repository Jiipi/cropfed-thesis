"""Run the full main-study matrix across FL scenarios.

Each scenario runs Flower Simulation against the real PlantVillage profiles.
Flower reads run_config from [tool.flwr.app.config] in pyproject.toml, so
this script rewrites that section per scenario, invokes Flower, and restores
the original section afterwards.

Usage:
    python scripts/run_main_study.py \\
        --output-root artifacts/main-study-seed2026 \\
        --rounds 10 --local-epochs 1 --num-cpus 4 \\
        [--only FL-IID-AVG [--only FL-A01-SCAF ...]]

    python scripts/run_main_study.py --list-scenarios --output-root /tmp/x
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
from cropfed.flower.smoke import SUPPORTED_ALGORITHMS, validate_run_artifacts  # noqa: E402

NUM_CLIENTS = 4
DEFAULT_PROXIMAL_MU = 0.01
DEFAULT_SCAFFOLD_SERVER_LR = 1.0
DEFAULT_MOON_TEMPERATURE = 0.5
DEFAULT_MOON_MU = 1.0
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
RUN_CONFIG_HEADER = "[tool.flwr.app.config]"

#: Profile shorthand used in scenario ids -> directory under --profiles-root.
PROFILE_DIRECTORIES: dict[str, str] = {
    "iid": "iid",
    "alpha-100": "dirichlet-alpha-100",
    "alpha-0.5": "dirichlet-alpha-0.5",
    "alpha-0.1": "dirichlet-alpha-0.1",
    "quantity-skew": "quantity-skew",
    "feature-skew": "feature-skew",
}


# The matrix answers the proposal's comparison questions in one pass:
#   * baselines (centralized, local-only) to frame every federated number;
#   * fedavg across four label-skew levels, to show degradation;
#   * the four non-IID-robust algorithms against the same alpha-0.1 profile,
#     so the algorithm is the only variable when they are compared;
#   * fedavg vs the best-known robust algorithm on quantity and feature skew,
#     so the two new skews are not merely partitioned but actually measured.
SCENARIOS: list[dict[str, object]] = [
    {"id": "CEN-MBV3", "mode": "centralized", "profile": "iid", "algorithm": "centralized"},
    {"id": "LOC-MBV3", "mode": "local-only", "profile": "alpha-0.5", "algorithm": "local-only"},
    {"id": "FL-IID-AVG", "mode": "federated", "profile": "iid", "algorithm": "fedavg"},
    {"id": "FL-A100-AVG", "mode": "federated", "profile": "alpha-100", "algorithm": "fedavg"},
    {"id": "FL-A05-AVG", "mode": "federated", "profile": "alpha-0.5", "algorithm": "fedavg"},
    {"id": "FL-A01-AVG", "mode": "federated", "profile": "alpha-0.1", "algorithm": "fedavg"},
    {"id": "FL-A05-PROX", "mode": "federated", "profile": "alpha-0.5", "algorithm": "fedprox"},
    {"id": "FL-A01-PROX", "mode": "federated", "profile": "alpha-0.1", "algorithm": "fedprox"},
    {"id": "FL-A01-BN", "mode": "federated", "profile": "alpha-0.1", "algorithm": "fedbn"},
    {"id": "FL-A01-SCAF", "mode": "federated", "profile": "alpha-0.1", "algorithm": "scaffold"},
    {"id": "FL-A01-MOON", "mode": "federated", "profile": "alpha-0.1", "algorithm": "moon"},
    {"id": "FL-QTY-AVG", "mode": "federated", "profile": "quantity-skew", "algorithm": "fedavg"},
    {"id": "FL-QTY-PROX", "mode": "federated", "profile": "quantity-skew", "algorithm": "fedprox"},
    {"id": "FL-FEAT-AVG", "mode": "federated", "profile": "feature-skew", "algorithm": "fedavg"},
    {"id": "FL-FEAT-BN", "mode": "federated", "profile": "feature-skew", "algorithm": "fedbn"},
]


def _resolve_profile_dir(profile: str, profiles_root: Path) -> Path:
    try:
        directory = PROFILE_DIRECTORIES[profile]
    except KeyError:
        raise ValueError(
            f"unknown profile: {profile}; expected one of "
            + ", ".join(sorted(PROFILE_DIRECTORIES))
        ) from None
    return profiles_root / directory


def _read_profile_metadata(profile_dir: Path) -> dict[str, object]:
    """Read partition metadata from the profile's own artifact.

    Deriving ``partition_kind`` from the directory name used to be enough when
    every non-IID profile was Dirichlet, but it silently mislabels the quantity
    and feature skews — a quantity-skew run would record itself as
    ``dirichlet`` with ``alpha=0``, contradicting its own partition summary and
    invalidating the protocol lock.  The artifact already states the truth.
    """

    profile_path = profile_dir / "profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"profile metadata not found: {profile_path}; regenerate the profile "
            "with `cropfed prepare-full-profiles`"
        )
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    partition_kind = str(document.get("partition_kind", ""))
    if partition_kind not in {"iid", "dirichlet", "quantity_skew", "feature_skew"}:
        raise ValueError(
            f"{profile_path} declares an unsupported partition_kind: {partition_kind!r}"
        )
    # A quantity-skew profile partitions labels IID and then reshapes the client
    # sizes, so its spec records partition_kind='iid' plus quantity_skew=True.
    # Reporting it as 'iid' would erase the skew from every artifact.
    if bool(document.get("quantity_skew", False)):
        partition_kind = "quantity_skew"
    alpha = document.get("dirichlet_alpha")
    return {
        "partition_kind": partition_kind,
        "dirichlet_alpha": float(alpha) if alpha is not None else None,
        "feature_skew_strength": document.get("feature_skew_strength"),
        "name": str(document.get("name", profile_dir.name)),
    }


def _algorithm_hyperparameters(
    algorithm: str,
    *,
    proximal_mu: float,
    scaffold_server_lr: float,
    moon_temperature: float,
    moon_mu: float,
) -> dict[str, float]:
    """Mirror the server's rule: a hyperparameter is 0.0 when not applicable.

    Kept identical to ``server_app._algorithm_hyperparameters`` so the values
    the launcher verifies are exactly the ones the run recorded.
    """

    algorithm = algorithm.lower()
    return {
        "proximal_mu": proximal_mu if algorithm == "fedprox" else 0.0,
        "scaffold_server_lr": scaffold_server_lr if algorithm == "scaffold" else 0.0,
        "moon_temperature": moon_temperature if algorithm == "moon" else 0.0,
        "moon_mu": moon_mu if algorithm == "moon" else 0.0,
    }


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
    metadata = _read_profile_metadata(profile_dir)
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
        partition_kind=str(metadata["partition_kind"]),
        dirichlet_alpha=metadata["dirichlet_alpha"],
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
    scaffold_server_lr: float,
    moon_temperature: float,
    moon_mu: float,
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
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"unsupported algorithm: {algorithm}; expected one of "
            + ", ".join(sorted(SUPPORTED_ALGORITHMS))
        )
    if algorithm in {"fedbn", "moon"} and rounds < 2:
        # Both need a previous round's local state, so a single round cannot
        # demonstrate them and the artifact validator would reject the run
        # after all the GPU time had been spent.
        raise ValueError(f"{algorithm} requires at least 2 rounds, got {rounds}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_profile_metadata(profile_dir)
    partition_kind = str(metadata["partition_kind"])
    dirichlet_alpha = metadata["dirichlet_alpha"]

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
        "scaffold-server-lr": scaffold_server_lr,
        "moon-temperature": moon_temperature,
        "moon-mu": moon_mu,
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
        hyperparameters=_algorithm_hyperparameters(
            algorithm,
            proximal_mu=proximal_mu,
            scaffold_server_lr=scaffold_server_lr,
            moon_temperature=moon_temperature,
            moon_mu=moon_mu,
        ),
    )
    validation["elapsed_seconds"] = elapsed
    validation["partition_kind"] = partition_kind
    validation["dirichlet_alpha"] = dirichlet_alpha
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
    scaffold_server_lr: float = DEFAULT_SCAFFOLD_SERVER_LR,
    moon_temperature: float = DEFAULT_MOON_TEMPERATURE,
    moon_mu: float = DEFAULT_MOON_MU,
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
            scaffold_server_lr=scaffold_server_lr,
            moon_temperature=moon_temperature,
            moon_mu=moon_mu,
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
    parser.add_argument("--output-root", type=Path)
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
    parser.add_argument(
        "--scaffold-server-lr", type=float, default=DEFAULT_SCAFFOLD_SERVER_LR
    )
    parser.add_argument(
        "--moon-temperature", type=float, default=DEFAULT_MOON_TEMPERATURE
    )
    parser.add_argument("--moon-mu", type=float, default=DEFAULT_MOON_MU)
    parser.add_argument("--research-run", action="store_true")
    parser.add_argument(
        "--protocol-lock-root",
        type=Path,
        help="directory containing one locked JSON file named <scenario-id>.json",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=[s["id"] for s in SCENARIOS],
        help="run only this scenario; repeatable to select several",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="print the scenario matrix and exit without running anything",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        for scenario in SCENARIOS:
            print(
                f"{scenario['id']:<14} {scenario['mode']:<12} "
                f"{scenario['algorithm']:<11} {scenario['profile']}"
            )
        return 0

    if args.research_run and args.protocol_lock_root is None:
        parser.error("--research-run requires --protocol-lock-root")
    if args.output_root is None:
        parser.error("--output-root is required unless --list-scenarios is given")
    if args.num_gpus < 0:
        parser.error("--num-gpus cannot be negative")
    if args.scaffold_server_lr <= 0:
        parser.error("--scaffold-server-lr must be positive")
    if args.moon_temperature <= 0:
        parser.error("--moon-temperature must be positive")
    if args.moon_mu < 0:
        parser.error("--moon-mu cannot be negative")

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
        # Keep matrix order rather than the order the flags were typed, so two
        # invocations of the same set produce comparable summaries.
        requested = set(args.only)
        selected = [s for s in SCENARIOS if s["id"] in requested]

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
                scaffold_server_lr=args.scaffold_server_lr,
                moon_temperature=args.moon_temperature,
                moon_mu=args.moon_mu,
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
    # A subset run is a legitimate way to re-run one scenario, but the summary
    # of a subset is not the main study and must not claim to be: the
    # comparison table it feeds would silently be missing rows.
    matrix_complete = len(selected) == len(SCENARIOS)
    research_valid = bool(args.research_run and all_succeeded and matrix_complete)
    summary = {
        "status": "completed" if all_succeeded else "failed",
        "result_kind": (
            "federated_image_research_candidate"
            if research_valid
            else "federated_image_main_study_pilot"
        ),
        "research_result_valid": research_valid,
        "matrix_complete": matrix_complete,
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
            "scaffold_server_lr": args.scaffold_server_lr,
            "moon_temperature": args.moon_temperature,
            "moon_mu": args.moon_mu,
            "model": args.model,
            "profiles_root": str(profiles_root),
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
        },
        "scenarios_selected": [str(s["id"]) for s in selected],
        "scenarios_available": [str(s["id"]) for s in SCENARIOS],
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
