"""Centralized image baseline using exactly the held-out test manifest."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
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


def run_centralized(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pretrained: bool,
    seed: int,
    output_dir: Path,
    research_result_valid: bool | None = None,
    protocol_lock: Path | None = None,
    class_names: Sequence[str] = TOMATO_CLASSES,
    class_groups: Sequence[str] = TOMATO_CLASS_GROUPS,
) -> dict[str, Any]:
    """Train on pooled data; this is the upper-bound comparison, not private FL."""

    input_manifests = {
        "train_sha256": file_sha256(train_manifest),
        "validation_sha256": file_sha256(validation_manifest),
        "test_sha256": file_sha256(test_manifest),
    }
    protocol_validation = None
    if research_result_valid is True:
        protocol_validation = validate_protocol_lock(
            protocol_lock,
            experiment_type="centralized",
            config={
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "pretrained": pretrained,
            },
            manifest_hashes=input_manifests,
            seed=seed,
        )
    output_dir = prepare_new_output_directory(output_dir)
    set_reproducible_seed(seed)
    device = select_device()
    resolved_class_names = tuple(class_names)
    resolved_class_groups = tuple(class_groups)
    model = build_model(
        model_name, num_classes=len(resolved_class_names), pretrained=pretrained
    )
    train_loader = build_dataloader(train_manifest, training=True, batch_size=batch_size)
    validation_loader = build_dataloader(
        validation_manifest, training=False, batch_size=batch_size
    )
    test_loader = build_dataloader(
        test_manifest, training=False, batch_size=batch_size
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
    evaluation = evaluate_model(
        model,
        test_loader,
        device=device,
        class_names=resolved_class_names,
        class_groups=resolved_class_groups,
    )
    elapsed = time.perf_counter() - started

    checkpoint = output_dir / "centralized_model.pt"
    checkpoint_info = save_model_checkpoint(
        checkpoint,
        model,
        model_name=model_name,
        metadata={
            "experiment_type": "centralized",
            "seed": seed,
            "epochs": epochs,
            "best_validation_epoch": train_result.best_epoch,
        },
        class_order=resolved_class_names,
    )
    result: dict[str, Any] = {
        "result_kind": (
            "image_baseline_pilot"
            if research_result_valid is False
            else "image_baseline_research_candidate"
            if research_result_valid is True
            else "image_baseline_unclassified"
        ),
        "experiment_type": "centralized",
        "model": model_name,
        "pretrained": pretrained,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_train": train_result.num_examples,
        "num_validation": train_result.best_validation.num_examples,
        "num_test": evaluation.num_examples,
        "train_loss": train_result.loss,
        "best_validation_epoch": train_result.best_epoch,
        "validation_loss": train_result.best_validation.loss,
        "validation_metrics": train_result.best_validation.metrics,
        "training_history": list(train_result.history),
        "metrics": evaluation.metrics,
        "test_loss": evaluation.loss,
        "elapsed_seconds": elapsed,
        "trainable_parameters": count_trainable_parameters(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "checkpoint_bytes": checkpoint_info["bytes"],
        "checkpoint_format_version": checkpoint_info["format_version"],
        "class_order": list(resolved_class_names),
        "input_manifests": input_manifests,
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
