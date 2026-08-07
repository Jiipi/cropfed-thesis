"""Generate protocol locks for the main-study scenarios.

A protocol lock is what separates a research result from an exploratory one:
``validate_protocol_lock`` refuses to let a run be marked research-valid unless
its config, its input manifest hashes, and its seed all match a lock file that
was written *before* the run.  That ordering is the whole point — a lock
generated from a finished run's own outputs would prove nothing.

The hard requirement is that a generated lock matches byte-for-byte what the
runner will present at validation time.  Rather than restate the config in a
second place, this script imports ``scripts/run_main_study.py`` and reuses its
``SCENARIOS`` matrix, its profile-metadata reader, and its argument defaults, so
a change to the study cannot silently invalidate every lock.

Usage::

    python scripts/generate_protocol_locks.py \\
        --output-root artifacts/protocol-locks-seed2026 \\
        --rounds 10 --local-epochs 1

Then run the study against them::

    python scripts/run_main_study.py --research-run \\
        --protocol-lock-root artifacts/protocol-locks-seed2026 ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cropfed.constants import taxonomy_from_scope  # noqa: E402
from cropfed.experiments.artifacts import file_sha256  # noqa: E402


def _load_main_study():
    """Import run_main_study.py by path.

    It is a script rather than a package module, and importing it is what keeps
    the scenario matrix single-sourced.
    """

    spec = importlib.util.spec_from_file_location(
        "run_main_study", PROJECT_ROOT / "scripts" / "run_main_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _centralized_config(*, model: str, epochs: int, batch_size: int,
                        learning_rate: float, pretrained: bool) -> dict[str, Any]:
    """Mirror the config dict built in ``run_centralized``."""

    return {
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
    }


def _local_only_config(*, model: str, epochs: int, batch_size: int,
                       learning_rate: float, pretrained: bool, num_clients: int,
                       partition_kind: str,
                       dirichlet_alpha: float | None) -> dict[str, Any]:
    """Mirror the config dict built in ``run_local_only``."""

    return {
        "model": model,
        "epochs_per_client": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "num_clients": num_clients,
        "partition_kind": partition_kind,
        "dirichlet_alpha": dirichlet_alpha,
    }


def _federated_config(*, algorithm: str, partition_kind: str,
                      dirichlet_alpha: float | None, model: str, num_clients: int,
                      num_rounds: int, local_epochs: int, batch_size: int,
                      learning_rate: float, pretrained: bool, proximal_mu: float,
                      scaffold_server_lr: float, moon_temperature: float,
                      moon_mu: float) -> dict[str, Any]:
    """Mirror the config dict built in ``server_app.main``.

    The four algorithm hyperparameters are recorded unconditionally and are zero
    when they do not apply, exactly as ``_algorithm_hyperparameters`` does; a
    lock that omitted them would fail validation for every scenario.
    """

    return {
        "algorithm": algorithm,
        "partition_kind": partition_kind,
        # The server reads dirichlet-alpha from run_config, where the runner
        # writes `dirichlet_alpha or 0.0`, then casts to float. An IID profile
        # therefore validates against 0.0, not None.
        "dirichlet_alpha": float(dirichlet_alpha or 0.0),
        "model": model,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "proximal_mu": proximal_mu if algorithm == "fedprox" else 0.0,
        "scaffold_server_lr": scaffold_server_lr if algorithm == "scaffold" else 0.0,
        "moon_temperature": moon_temperature if algorithm == "moon" else 0.0,
        "moon_mu": moon_mu if algorithm == "moon" else 0.0,
    }


def build_lock(
    scenario: dict[str, Any],
    *,
    study,
    profiles_root: Path,
    allowed_seeds: list[int],
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    model: str,
    pretrained: bool,
    proximal_mu: float,
    scaffold_server_lr: float,
    moon_temperature: float,
    moon_mu: float,
) -> dict[str, Any]:
    """Build one lock document for one scenario."""

    mode = str(scenario["mode"])
    profile_dir = study._resolve_profile_dir(str(scenario["profile"]), profiles_root)
    if not profile_dir.is_dir():
        raise FileNotFoundError(
            f"profile directory not found: {profile_dir}; prepare the profiles "
            "before generating locks, since the lock records their hashes"
        )
    metadata = study._read_profile_metadata(profile_dir)
    partition_kind = str(metadata["partition_kind"])
    dirichlet_alpha = metadata["dirichlet_alpha"]

    if mode == "centralized":
        experiment_type = "centralized"
        config = _centralized_config(
            model=model,
            epochs=rounds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pretrained=pretrained,
        )
        manifest_hashes = {
            "train_sha256": file_sha256(profile_dir / "pooled_train_manifest.csv"),
            "validation_sha256": file_sha256(profile_dir / "validation_manifest.csv"),
            "test_sha256": file_sha256(profile_dir / "test_manifest.csv"),
        }
    elif mode == "local-only":
        experiment_type = "local-only"
        config = _local_only_config(
            model=model,
            epochs=rounds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pretrained=pretrained,
            num_clients=study.NUM_CLIENTS,
            partition_kind=partition_kind,
            dirichlet_alpha=dirichlet_alpha,
        )
        manifest_hashes = _federated_manifest_hashes(profile_dir)
    elif mode == "federated":
        experiment_type = "federated"
        config = _federated_config(
            algorithm=str(scenario["algorithm"]),
            partition_kind=partition_kind,
            dirichlet_alpha=dirichlet_alpha,
            model=model,
            num_clients=study.NUM_CLIENTS,
            num_rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pretrained=pretrained,
            proximal_mu=proximal_mu,
            scaffold_server_lr=scaffold_server_lr,
            moon_temperature=moon_temperature,
            moon_mu=moon_mu,
        )
        manifest_hashes = _federated_manifest_hashes(profile_dir)
    else:
        raise ValueError(f"unknown scenario mode: {mode}")

    return {
        "schema_version": 1,
        "status": "locked",
        "experiment_type": experiment_type,
        "scenario_id": str(scenario["id"]),
        "profile": str(scenario["profile"]),
        "config": config,
        "manifest_hashes": manifest_hashes,
        "allowed_seeds": allowed_seeds,
    }


def _federated_manifest_hashes(profile_dir: Path) -> dict[str, str | None]:
    """Hashes for the two artifacts a federated or local-only run reads.

    ``partition_summary`` may legitimately be absent, and the consumer records
    ``None`` in that case; the lock must say the same thing rather than omit the
    key, or the comparison fails on a missing entry.
    """

    partition_summary = profile_dir / "clients" / "partition_summary.json"
    return {
        "global_test_sha256": file_sha256(profile_dir / "test_manifest.csv"),
        "partition_summary_sha256": (
            file_sha256(partition_summary) if partition_summary.is_file() else None
        ),
    }


def generate_locks(
    output_root: Path,
    *,
    profiles_root: Path,
    allowed_seeds: list[int],
    only: list[str] | None = None,
    **options: Any,
) -> dict[str, Any]:
    study = _load_main_study()
    scenarios = study.SCENARIOS
    if only:
        requested = set(only)
        unknown = requested - {str(s["id"]) for s in scenarios}
        if unknown:
            raise ValueError(f"unknown scenario ids: {sorted(unknown)}")
        scenarios = [s for s in scenarios if str(s["id"]) in requested]

    output_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for scenario in scenarios:
        lock = build_lock(
            scenario,
            study=study,
            profiles_root=profiles_root,
            allowed_seeds=allowed_seeds,
            **options,
        )
        # The runner looks for <scenario-id>.json in lower case.
        destination = output_root / f"{str(scenario['id']).lower()}.json"
        destination.write_text(
            json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(
            {
                "scenario_id": lock["scenario_id"],
                "experiment_type": lock["experiment_type"],
                "lock": destination.name,
                "sha256": file_sha256(destination),
            }
        )

    return {
        "schema_version": 1,
        "status": "locked",
        "profiles_root": profiles_root.as_posix(),
        "allowed_seeds": allowed_seeds,
        "num_locks": len(written),
        "locks": written,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="write one protocol lock per main-study scenario"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "flower-profiles-full",
    )
    parser.add_argument(
        "--allowed-seed",
        type=int,
        action="append",
        help="repeatable; defaults to 2026 2027 2028 for three-seed repeats",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--model",
        choices=["mobilenet_v3_small", "efficientnet_lite0", "mobilenet_v2"],
        default="mobilenet_v3_small",
    )
    parser.add_argument("--taxonomy", default="plantvillage-full")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--scaffold-server-lr", type=float, default=1.0)
    parser.add_argument("--moon-temperature", type=float, default=0.5)
    parser.add_argument("--moon-mu", type=float, default=1.0)
    parser.add_argument("--only", action="append", help="repeatable scenario id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    taxonomy_from_scope(args.taxonomy)  # Reject an unknown scope early.
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            f"refusing to overwrite existing locks: {output_root}; a lock must be "
            "written before the run it governs, so replacing one silently would "
            "defeat the purpose"
        )
    report = generate_locks(
        output_root,
        profiles_root=args.profiles_root.expanduser().resolve(),
        allowed_seeds=sorted(set(args.allowed_seed or [2026, 2027, 2028])),
        only=args.only,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model=args.model,
        pretrained=not args.no_pretrained,
        proximal_mu=args.proximal_mu,
        scaffold_server_lr=args.scaffold_server_lr,
        moon_temperature=args.moon_temperature,
        moon_mu=args.moon_mu,
    )
    summary = output_root / "protocol_locks_summary.json"
    summary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
