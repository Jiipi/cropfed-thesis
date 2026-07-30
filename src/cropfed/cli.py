"""Command-line entry points for safe project bootstrapping."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cropfed.config import ExperimentConfig
from cropfed.data.manifest import (
    content_grouped_stratified_train_test_split,
    scan_plantvillage_tomato,
    write_client_manifests,
    write_manifest,
)
from cropfed.simulation import run_synthetic_experiment


def _configure_utf8_stream(stream: object) -> None:
    """Keep Vietnamese CLI output usable on legacy Windows code pages."""

    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    except (OSError, ValueError):
        # Captured/closed streams used by callers may not be reconfigurable.
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cropfed")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="run a dependency-light synthetic FL smoke test"
    )
    demo.add_argument("--algorithm", choices=["fedavg", "fedprox"], default="fedavg")
    demo.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    demo.add_argument("--alpha", type=float, default=0.5)
    demo.add_argument("--clients", type=int, default=4)
    demo.add_argument("--rounds", type=int, default=5)
    demo.add_argument("--local-epochs", type=int, default=2)
    demo.add_argument("--learning-rate", type=float, default=0.05)
    demo.add_argument("--proximal-mu", type=float, default=0.01)
    demo.add_argument("--seed", type=int, default=2026)
    demo.add_argument("--output", type=Path, default=Path("artifacts/smoke-result.json"))

    prepare = subparsers.add_parser(
        "prepare-data", help="scan PlantVillage tomato folders and write manifests"
    )
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument(
        "--client-output-root",
        type=Path,
        help="defaults to a sibling 'clients' directory next to --output-dir",
    )
    prepare.add_argument("--test-fraction", type=float, default=0.2)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument("--clients", type=int, default=4)
    prepare.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    prepare.add_argument("--alpha", type=float, default=0.5)
    prepare.add_argument("--seed", type=int, default=2026)

    prepare_profiles = subparsers.add_parser(
        "prepare-mvp-profiles",
        help="create and audit IID, Non-IID alpha=0.5, and Non-IID alpha=0.1",
    )
    prepare_profiles.add_argument("--dataset-root", type=Path, required=True)
    prepare_profiles.add_argument(
        "--output-root", type=Path, default=Path("data/mvp_profiles")
    )
    prepare_profiles.add_argument("--test-fraction", type=float, default=0.2)
    prepare_profiles.add_argument("--validation-fraction", type=float, default=0.2)
    prepare_profiles.add_argument("--clients", type=int, default=4)
    prepare_profiles.add_argument("--seed", type=int, default=2026)

    audit = subparsers.add_parser(
        "audit-data",
        help="verify prepared images, manifests, split isolation, and client assignment",
    )
    audit.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/processed/train_manifest.csv"),
    )
    audit.add_argument(
        "--test-manifest",
        type=Path,
        default=Path("data/processed/test_manifest.csv"),
    )
    audit.add_argument(
        "--client-data-root",
        type=Path,
        default=Path("data/clients"),
    )
    audit.add_argument("--clients", type=int, default=4)
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/data_audit.json"),
    )

    centralized = subparsers.add_parser(
        "train-centralized", help="train the centralized MobileNetV2 baseline"
    )
    centralized.add_argument(
        "--train-manifest", type=Path, default=Path("data/processed/train_manifest.csv")
    )
    centralized.add_argument(
        "--test-manifest", type=Path, default=Path("data/processed/test_manifest.csv")
    )
    centralized.add_argument(
        "--model",
        choices=["mobilenet_v2", "resnet18"],
        default="mobilenet_v2",
    )
    centralized.add_argument("--epochs", type=int, default=10)
    centralized.add_argument("--batch-size", type=int, default=32)
    centralized.add_argument("--learning-rate", type=float, default=0.001)
    centralized.add_argument("--seed", type=int, default=2026)
    centralized.add_argument("--no-pretrained", action="store_true")
    centralized.add_argument(
        "--pilot",
        action="store_true",
        help="mark this short validation run as ineligible for research export",
    )
    centralized.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/centralized")
    )

    local_only = subparsers.add_parser(
        "train-local-only",
        help="train one independent MobileNetV2 per agricultural client",
    )
    local_only.add_argument(
        "--client-data-root", type=Path, default=Path("data/clients")
    )
    local_only.add_argument(
        "--test-manifest", type=Path, default=Path("data/processed/test_manifest.csv")
    )
    local_only.add_argument("--clients", type=int, default=4)
    local_only.add_argument(
        "--model", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2"
    )
    local_only.add_argument("--epochs", type=int, default=30)
    local_only.add_argument("--batch-size", type=int, default=32)
    local_only.add_argument("--learning-rate", type=float, default=0.001)
    local_only.add_argument("--seed", type=int, default=2026)
    local_only.add_argument(
        "--partition", choices=["iid", "dirichlet"], default="dirichlet"
    )
    local_only.add_argument("--alpha", type=float, default=0.5)
    local_only.add_argument("--no-pretrained", action="store_true")
    local_only.add_argument(
        "--pilot",
        action="store_true",
        help="mark this short validation run as ineligible for research export",
    )
    local_only.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/local-only")
    )

    export = subparsers.add_parser(
        "export-results",
        help="export research-candidate runs; synthetic smoke is always excluded",
    )
    export.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="repeat for each centralized/local-only/Flower run directory",
    )
    export.add_argument("--output-dir", type=Path, required=True)

    predict = subparsers.add_parser(
        "predict", help="classify one image locally without uploading it"
    )
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument(
        "--model",
        choices=["mobilenet_v2", "resnet18"],
        default=None,
        help="inferred from versioned checkpoints; required for legacy ResNet18",
    )
    predict.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_stream(sys.stdout)
    _configure_utf8_stream(sys.stderr)
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        config = ExperimentConfig(
            algorithm=args.algorithm,
            partition_kind=args.partition,
            num_clients=args.clients,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            dirichlet_alpha=args.alpha,
            proximal_mu=args.proximal_mu,
            seed=args.seed,
        )
        result = run_synthetic_experiment(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        final = result["final_metrics"]
        print(
            f"synthetic smoke complete: accuracy={final['accuracy']:.4f}, "
            f"macro_f1={final['macro_f1']:.4f}, output={args.output}"
        )
        return 0

    if args.command == "prepare-data":
        records = scan_plantvillage_tomato(args.dataset_root)
        train, test, split_statistics = content_grouped_stratified_train_test_split(
            records, test_fraction=args.test_fraction, seed=args.seed
        )
        write_manifest(train, args.output_dir / "train_manifest.csv")
        write_manifest(test, args.output_dir / "test_manifest.csv")
        write_client_manifests(
            train,
            args.client_output_root or (args.output_dir.parent / "clients"),
            num_clients=args.clients,
            partition_kind=args.partition,
            alpha=args.alpha,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        print(
            f"prepared manifests: train={len(train)}, test={len(test)}, "
            f"clients={args.clients}, duplicate_groups="
            f"{split_statistics['num_duplicate_content_groups']}, "
            f"output={args.output_dir}"
        )
        return 0

    if args.command == "prepare-mvp-profiles":
        from cropfed.data.profiles import prepare_mvp_profiles

        result = prepare_mvp_profiles(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            num_clients=args.clients,
            seed=args.seed,
        )
        invariants = result["shared_split_invariants"]
        print(
            f"MVP profiles {result['status']}: profiles={len(result['profiles'])}, "
            f"train={result['num_train']}, test={result['num_test']}, "
            f"shared_test={invariants['same_global_test_manifest']}, "
            f"output={args.output_root}"
        )
        return 0 if result["status"] == "passed" else 2

    if args.command == "audit-data":
        from cropfed.data.audit import audit_prepared_data, write_audit_report

        report = audit_prepared_data(
            train_manifest=args.train_manifest,
            test_manifest=args.test_manifest,
            client_data_root=args.client_data_root,
            num_clients=args.clients,
        )
        write_audit_report(report, args.output)
        print(
            f"data audit {report['status']}: "
            f"images={report['images']['unique_paths_checked']}, "
            f"errors={len(report['errors'])}, warnings={len(report['warnings'])}, "
            f"output={args.output}"
        )
        return 0 if report["status"] == "passed" else 2

    if args.command == "train-centralized":
        from cropfed.experiments.centralized import run_centralized

        result = run_centralized(
            train_manifest=args.train_manifest,
            test_manifest=args.test_manifest,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pretrained=not args.no_pretrained,
            seed=args.seed,
            output_dir=args.output_dir,
            research_result_valid=False if args.pilot else None,
        )
        print(
            f"centralized complete: accuracy={result['metrics']['accuracy']:.4f}, "
            f"macro_f1={result['metrics']['macro_f1']:.4f}, output={args.output_dir}"
        )
        return 0

    if args.command == "train-local-only":
        from cropfed.experiments.local_only import run_local_only

        result = run_local_only(
            client_data_root=args.client_data_root,
            test_manifest=args.test_manifest,
            num_clients=args.clients,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pretrained=not args.no_pretrained,
            seed=args.seed,
            output_dir=args.output_dir,
            partition_kind=args.partition,
            dirichlet_alpha=args.alpha if args.partition == "dirichlet" else None,
            research_result_valid=False if args.pilot else None,
        )
        print(
            "local-only complete: "
            f"mean_global_macro_f1={result['summary']['mean_global_macro_f1']:.4f}, "
            f"worst_global_macro_f1={result['summary']['worst_global_macro_f1']:.4f}"
        )
        return 0

    if args.command == "export-results":
        from cropfed.experiments.export import export_results

        result = export_results(
            args.run,
            args.output_dir,
            project_root=Path.cwd(),
        )
        print(
            f"export complete: included={len(result['included'])}, "
            f"excluded={len(result['excluded'])}, output={args.output_dir}"
        )
        return 0

    if args.command == "predict":
        from cropfed.ml.inference import predict_image

        result = predict_image(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            model_name=args.model,
            top_k=args.top_k,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
