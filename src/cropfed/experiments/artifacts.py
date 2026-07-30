"""Shared artifact safeguards for centralized and local-only baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cropfed.experiments.export import build_environment_manifest


def prepare_new_output_directory(output_dir: Path) -> Path:
    """Create a run directory and refuse to overwrite any existing artifact."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_environment_artifact(output_dir: Path) -> dict[str, object]:
    destination = output_dir / "environment.json"
    destination.write_text(
        json.dumps(
            build_environment_manifest(Path.cwd()),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "path": destination.name,
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
    }
