"""Fail-closed validation for protocol-locked research runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def validate_protocol_lock(
    lock_path: Path | None,
    *,
    experiment_type: str,
    config: dict[str, Any],
    manifest_hashes: dict[str, str | None],
    seed: int,
) -> dict[str, Any]:
    """Validate a lock artifact against the exact run before research export."""

    if lock_path is None:
        raise ValueError("research runs require --protocol-lock")
    resolved = lock_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"protocol lock does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol lock must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("protocol lock schema_version must be 1")
    if payload.get("status") != "locked":
        raise ValueError("protocol lock status must be 'locked'")
    if payload.get("experiment_type") != experiment_type:
        raise ValueError("protocol lock experiment_type does not match the run")

    locked_config = payload.get("config")
    if not isinstance(locked_config, dict):
        raise ValueError("protocol lock config must be an object")
    config_mismatches = {
        key: {"locked": locked_config.get(key), "actual": value}
        for key, value in config.items()
        if locked_config.get(key) != value
    }
    if config_mismatches:
        raise ValueError(f"run config differs from protocol lock: {config_mismatches}")

    locked_manifests = payload.get("manifest_hashes")
    if not isinstance(locked_manifests, dict):
        raise ValueError("protocol lock manifest_hashes must be an object")
    manifest_mismatches = {
        key: {"locked": locked_manifests.get(key), "actual": value}
        for key, value in manifest_hashes.items()
        if locked_manifests.get(key) != value
    }
    if manifest_mismatches:
        raise ValueError(
            f"input manifests differ from protocol lock: {manifest_mismatches}"
        )

    allowed_seeds = payload.get("allowed_seeds")
    if (
        not isinstance(allowed_seeds, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in allowed_seeds)
        or seed not in allowed_seeds
    ):
        raise ValueError("run seed is not listed in protocol lock allowed_seeds")

    return {
        "schema_version": 1,
        "status": "locked",
        "filename": resolved.name,
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
