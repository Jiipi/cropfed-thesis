"""Prepare the three reproducible MVP data-distribution profiles.

The global train/test split is created exactly once.  Only the training split
is repartitioned, which keeps the held-out test set identical across IID,
moderate Non-IID, and strong Non-IID experiments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cropfed.data.audit import audit_prepared_data, write_audit_report
from cropfed.data.manifest import (
    content_grouped_stratified_train_test_split,
    scan_plantvillage_tomato,
    write_client_manifests,
    write_manifest,
)


@dataclass(frozen=True, slots=True)
class DataProfileSpec:
    """Stable identifier and partition parameters for one MVP profile."""

    name: str
    partition_kind: str
    dirichlet_alpha: float | None


MVP_PROFILE_SPECS: tuple[DataProfileSpec, ...] = (
    DataProfileSpec("iid", "iid", None),
    DataProfileSpec("dirichlet-alpha-0.5", "dirichlet", 0.5),
    DataProfileSpec("dirichlet-alpha-0.1", "dirichlet", 0.1),
)


def prepare_mvp_profiles(
    *,
    dataset_root: Path,
    output_root: Path,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    num_clients: int = 4,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create and audit IID/Non-IID profiles with one shared global test set.

    ``output_root`` must be empty so a prior experimental partition cannot be
    overwritten accidentally.  Every profile receives byte-identical master
    manifests and a different ``clients`` directory.
    """

    output_root = _prepare_empty_output_root(output_root)
    records = scan_plantvillage_tomato(dataset_root)
    train_records, test_records, split_statistics = (
        content_grouped_stratified_train_test_split(
        records,
        test_fraction=test_fraction,
        seed=seed,
        )
    )

    profile_rows: list[dict[str, Any]] = []
    for spec in MVP_PROFILE_SPECS:
        profile_root = output_root / spec.name
        train_manifest = profile_root / "train_manifest.csv"
        test_manifest = profile_root / "test_manifest.csv"
        client_root = profile_root / "clients"
        write_manifest(train_records, train_manifest)
        write_manifest(test_records, test_manifest)
        write_client_manifests(
            train_records,
            client_root,
            num_clients=num_clients,
            partition_kind=spec.partition_kind,
            alpha=spec.dirichlet_alpha or 0.5,
            validation_fraction=validation_fraction,
            seed=seed,
        )

        audit = audit_prepared_data(
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            client_data_root=client_root,
            num_clients=num_clients,
        )
        audit_path = profile_root / "data_audit.json"
        write_audit_report(audit, audit_path)
        profile_metadata = {
            **asdict(spec),
            "num_clients": num_clients,
            "seed": seed,
            "num_train": len(train_records),
            "num_test": len(test_records),
            "test_fraction": test_fraction,
            "validation_fraction": validation_fraction,
            "train_manifest_sha256": _sha256_file(train_manifest),
            "test_manifest_sha256": _sha256_file(test_manifest),
            "partition_summary_sha256": _sha256_file(
                client_root / "partition_summary.json"
            ),
            "data_audit_sha256": _sha256_file(audit_path),
            "audit_status": audit["status"],
            "paths": {
                "train_manifest": str(train_manifest.relative_to(output_root)),
                "test_manifest": str(test_manifest.relative_to(output_root)),
                "client_data_root": str(client_root.relative_to(output_root)),
                "data_audit": str(audit_path.relative_to(output_root)),
            },
        }
        profile_path = profile_root / "profile.json"
        profile_path.write_text(
            json.dumps(profile_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        profile_rows.append(
            {
                **profile_metadata,
                "profile_sha256": _sha256_file(profile_path),
            }
        )

    test_hashes = {row["test_manifest_sha256"] for row in profile_rows}
    train_hashes = {row["train_manifest_sha256"] for row in profile_rows}
    all_audits_passed = all(row["audit_status"] == "passed" for row in profile_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "profile_set": "tomato_mvp_iid_non_iid",
        "status": "passed" if all_audits_passed else "failed",
        "seed": seed,
        "num_clients": num_clients,
        "num_source_images": len(records),
        "num_train": len(train_records),
        "num_test": len(test_records),
        "content_grouped_split": split_statistics,
        "shared_split_invariants": {
            "same_train_manifest": len(train_hashes) == 1,
            "same_global_test_manifest": len(test_hashes) == 1,
            "train_manifest_sha256": next(iter(train_hashes)),
            "test_manifest_sha256": next(iter(test_hashes)),
        },
        "profiles": profile_rows,
    }
    (output_root / "profiles_index.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _prepare_empty_output_root(output_root: Path) -> Path:
    resolved = output_root.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty profile set: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
