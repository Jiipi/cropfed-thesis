"""Centralized image baseline using exactly the held-out test manifest."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cropfed.constants import TOMATO_CLASSES
from cropfed.data.torch_data import build_dataloader
from cropfed.experiments.artifacts import (
    file_sha256,
    prepare_new_output_directory,
    write_environment_artifact,
)
from cropfed.ml.checkpoint import save_model_checkpoint
from cropfed.ml.model import build_model, count_trainable_parameters
from cropfed.ml.trainer import evaluate_model, select_device, set_reproducible_seed, train_local


def run_centralized(
    *,
    train_manifest: Path,
    test_manifest: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pretrained: bool,
    seed: int,
    output_dir: Path,
    research_result_valid: bool | None = None,
) -> dict[str, Any]:
    """Train on pooled data; this is the upper-bound comparison, not private FL."""

    output_dir = prepare_new_output_directory(output_dir)
    set_reproducible_seed(seed)
    device = select_device()
    model = build_model(
        model_name, num_classes=len(TOMATO_CLASSES), pretrained=pretrained
    )
    train_loader = build_dataloader(
        train_manifest, training=True, batch_size=batch_size
    )
    test_loader = build_dataloader(
        test_manifest, training=False, batch_size=batch_size
    )
    started = time.perf_counter()
    train_result = train_local(
        model,
        train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    evaluation = evaluate_model(model, test_loader, device=device)
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
        },
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
        "num_test": evaluation.num_examples,
        "train_loss": train_result.loss,
        "metrics": evaluation.metrics,
        "test_loss": evaluation.loss,
        "elapsed_seconds": elapsed,
        "trainable_parameters": count_trainable_parameters(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "checkpoint_bytes": checkpoint_info["bytes"],
        "checkpoint_format_version": checkpoint_info["format_version"],
        "class_order": list(TOMATO_CLASSES),
        "input_manifests": {
            "train_sha256": file_sha256(train_manifest),
            "test_sha256": file_sha256(test_manifest),
        },
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
