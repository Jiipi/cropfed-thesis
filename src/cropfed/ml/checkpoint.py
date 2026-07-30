"""Versioned, taxonomy-aware model checkpoint helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cropfed.constants import PROJECT_VERSION, TOMATO_CLASSES

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_KIND = "cropfed_image_classifier"


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    state_dict: Mapping[str, Any]
    format_version: int
    model_name: str | None
    model_version: str
    class_order: tuple[str, ...] | None
    metadata: dict[str, Any]


def save_model_checkpoint(
    destination: Path,
    model,
    *,
    model_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save CPU weights with the class contract and reproducibility metadata."""

    import torch

    if not model_name:
        raise ValueError("model_name cannot be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    payload = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_version": PROJECT_VERSION,
        "model_name": model_name,
        "class_order": list(TOMATO_CLASSES),
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
        "state_dict": state_dict,
    }
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return {
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "bytes": destination.stat().st_size,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_version": PROJECT_VERSION,
        "model_name": model_name,
    }


def load_model_checkpoint(path: Path) -> LoadedCheckpoint:
    """Load the current envelope or a legacy raw state dict safely on CPU."""

    import torch

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and payload.get("checkpoint_kind") == CHECKPOINT_KIND:
        format_version = int(payload.get("format_version", -1))
        if format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format version: {format_version}"
            )
        state_dict = payload.get("state_dict")
        if not _looks_like_state_dict(state_dict):
            raise ValueError("checkpoint state_dict is missing or invalid")
        class_order = tuple(str(item) for item in payload.get("class_order", ()))
        if class_order != TOMATO_CLASSES:
            raise ValueError("checkpoint class order does not match project taxonomy")
        return LoadedCheckpoint(
            state_dict=state_dict,
            format_version=format_version,
            model_name=str(payload["model_name"]),
            model_version=str(payload.get("model_version", "unknown")),
            class_order=class_order,
            metadata=dict(payload.get("metadata") or {}),
        )
    if _looks_like_state_dict(payload):
        return LoadedCheckpoint(
            state_dict=payload,
            format_version=0,
            model_name=None,
            model_version="legacy",
            class_order=None,
            metadata={},
        )
    raise ValueError("file is not a supported CropFed checkpoint")


def _looks_like_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(name, str) and hasattr(tensor, "shape")
            for name, tensor in value.items()
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
