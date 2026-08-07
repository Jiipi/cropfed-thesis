"""Local-only baseline: no collaboration and no parameter exchange."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from cropfed.constants import TOMATO_CLASS_GROUPS, TOMATO_CLASSES
from cropfed.data.torch_data import build_dataloader
from cropfed.experiments.artifacts import (
    file_sha256,
    prepare_new_output_directory,
    write_environment_artifact,
)
from cropfed.experiments.protocol import validate_protocol_lock
from cropfed.ml.checkpoint import save_model_checkpoint
from cropfed.ml.model import build_model, count_trainable_parameters
from cropfed.ml.trainer import (
    evaluate_model,
    select_device,
    set_reproducible_seed,
    train_with_validation,
)


def run_local_only(
    *,
    client_data_root: Path,
    test_manifest: Path,
    num_clients: int,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pretrained: bool,
    seed: int,
    output_dir: Path,
    partition_kind: str = "unspecified",
    dirichlet_alpha: float | None = None,
    research_result_valid: bool | None = None,
    protocol_lock: Path | None = None,
    class_names: Sequence[str] = TOMATO_CLASSES,
    class_groups: Sequence[str] = TOMATO_CLASS_GROUPS,
    dataset_root: Path | str | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train independent client models from identical seeded initialization."""

    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    partition_summary = client_data_root / "partition_summary.json"
    input_manifests = {
        "global_test_sha256": file_sha256(test_manifest),
        "partition_summary_sha256": (
            file_sha256(partition_summary) if partition_summary.is_file() else None
        ),
    }
    protocol_validation = None
    if research_result_valid is True:
        protocol_validation = validate_protocol_lock(
            protocol_lock,
            experiment_type="local-only",
            config={
                "model": model_name,
                "epochs_per_client": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "pretrained": pretrained,
                "num_clients": num_clients,
                "partition_kind": partition_kind,
                "dirichlet_alpha": dirichlet_alpha,
            },
            manifest_hashes=input_manifests,
            seed=seed,
        )
    output_dir = prepare_new_output_directory(output_dir)
    resolved_class_names = tuple(class_names)
    resolved_class_groups = tuple(class_groups)
    device = select_device()
    client_results: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for client_id in range(num_clients):
        train_manifest = (
            client_data_root / f"client_{client_id}" / "train_manifest.csv"
        )
        validation_manifest = (
            client_data_root / f"client_{client_id}" / "val_manifest.csv"
        )
        if not train_manifest.is_file() or not validation_manifest.is_file():
            raise FileNotFoundError(f"missing manifests for client {client_id}")

        # Reset before every build so all clients start from the same classifier weights.
        set_reproducible_seed(seed)
        model = build_model(
            model_name,
            num_classes=len(resolved_class_names),
            pretrained=pretrained,
        )
        train_loader = build_dataloader(
            train_manifest,
            training=True,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_root=dataset_root,
        )
        validation_loader = build_dataloader(
            validation_manifest,
            training=False,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_root=dataset_root,
        )
        global_test_loader = build_dataloader(
            test_manifest,
            training=False,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_root=dataset_root,
        )

        started = time.perf_counter()
        train_result = train_with_validation(
            model,
            train_loader,
            validation_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            class_names=resolved_class_names,
            class_groups=resolved_class_groups,
        )
        global_test = evaluate_model(
            model,
            global_test_loader,
            device=device,
            class_names=resolved_class_names,
            class_groups=resolved_class_groups,
        )
        elapsed = time.perf_counter() - started
        checkpoint = output_dir / f"client_{client_id}_model.pt"
        checkpoint_info = save_model_checkpoint(
            checkpoint,
            model,
            model_name=model_name,
            metadata={
                "experiment_type": "local-only",
                "client_id": client_id,
                "seed": seed,
                "epochs": epochs,
                "best_validation_epoch": train_result.best_epoch,
            },
            class_order=resolved_class_names,
        )

        client_results.append(
            {
                "client_id": client_id,
                "num_train": train_result.num_examples,
                "train_loss": train_result.loss,
                "best_validation_epoch": train_result.best_epoch,
                "training_history": list(train_result.history),
                "local_validation_loss": train_result.best_validation.loss,
                "local_validation_metrics": train_result.best_validation.metrics,
                "global_test_loss": global_test.loss,
                "global_test_metrics": global_test.metrics,
                "elapsed_seconds": elapsed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_info["sha256"],
                "checkpoint_bytes": checkpoint_info["bytes"],
                "checkpoint_format_version": checkpoint_info["format_version"],
                "input_manifests": {
                    "train_sha256": file_sha256(train_manifest),
                    "validation_sha256": file_sha256(validation_manifest),
                    "global_test_sha256": file_sha256(test_manifest),
                },
            }
        )

    global_f1 = [
        float(result["global_test_metrics"]["macro_f1"])
        for result in client_results
    ]
    global_accuracy = [
        float(result["global_test_metrics"]["accuracy"])
        for result in client_results
    ]
    local_f1 = [
        float(result["local_validation_metrics"]["macro_f1"])
        for result in client_results
    ]
    result: dict[str, Any] = {
        "result_kind": (
            "image_baseline_pilot"
            if research_result_valid is False
            else "image_baseline_research_candidate"
            if research_result_valid is True
            else "image_baseline_unclassified"
        ),
        "experiment_type": "local-only",
        "model": model_name,
        "pretrained": pretrained,
        "seed": seed,
        "partition_kind": partition_kind,
        "dirichlet_alpha": dirichlet_alpha,
        "num_clients": num_clients,
        "epochs_per_client": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trainable_parameters": count_trainable_parameters(model),
        "summary": {
            "mean_global_accuracy": mean(global_accuracy),
            "mean_global_macro_f1": mean(global_f1),
            "worst_global_macro_f1": min(global_f1),
            "best_global_macro_f1": max(global_f1),
            "mean_local_validation_macro_f1": mean(local_f1),
            "elapsed_seconds": time.perf_counter() - run_started,
        },
        "clients": client_results,
        "class_order": list(resolved_class_names),
        "global_test_manifest_sha256": input_manifests["global_test_sha256"],
        "partition_summary_sha256": input_manifests["partition_summary_sha256"],
        "protocol_lock": protocol_validation,
        "environment": write_environment_artifact(output_dir),
        "research_result_valid": research_result_valid,
        "research_validation_status": (
            "pilot_not_for_research"
            if research_result_valid is False
            else "validated_research_candidate"
            if research_result_valid is True
            else "not_declared"
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
