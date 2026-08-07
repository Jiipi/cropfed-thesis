"""Prepare reproducible PlantVillage data-distribution profiles.

The global train/test split is created exactly once.  Only the training split
is repartitioned, which keeps the held-out test set identical across IID,
moderate Non-IID, and strong Non-IID experiments.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cropfed.constants import (
    PLANTVILLAGE_FULL_TAXONOMY,
    TOMATO_TAXONOMY,
    DatasetTaxonomy,
)
from cropfed.data.audit import audit_prepared_data, write_audit_report
from cropfed.data.manifest import (
    content_grouped_stratified_train_test_split,
    read_manifest,
    scan_plantvillage,
    write_client_manifests,
    write_manifest,
)


@dataclass(frozen=True, slots=True)
class DataProfileSpec:
    """Stable identifier and partition parameters for one data profile."""

    name: str
    partition_kind: str
    dirichlet_alpha: float | None
    quantity_skew: bool = False
    feature_skew_strength: float = 0.5


MVP_PROFILE_SPECS: tuple[DataProfileSpec, ...] = (
    DataProfileSpec("iid", "iid", None),
    DataProfileSpec("dirichlet-alpha-0.5", "dirichlet", 0.5),
    DataProfileSpec("dirichlet-alpha-0.1", "dirichlet", 0.1),
)

# The proposal describes alpha≈100 as the near-IID control and alpha=0.1 as
# strong label skew.  Exact IID is retained as an additional sanity control.
FULL_PROFILE_SPECS: tuple[DataProfileSpec, ...] = (
    DataProfileSpec("iid", "iid", None),
    DataProfileSpec("dirichlet-alpha-100", "dirichlet", 100.0),
    DataProfileSpec("dirichlet-alpha-0.5", "dirichlet", 0.5),
    DataProfileSpec("dirichlet-alpha-0.1", "dirichlet", 0.1),
    DataProfileSpec("quantity-skew", "iid", None, quantity_skew=True),
    DataProfileSpec("feature-skew", "feature_skew", None),
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
    """Create the backward-compatible ten-class tomato pilot profiles."""

    return prepare_data_profiles(
        dataset_root=dataset_root,
        output_root=output_root,
        taxonomy=TOMATO_TAXONOMY,
        profile_specs=MVP_PROFILE_SPECS,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        num_clients=num_clients,
        seed=seed,
    )


def prepare_full_profiles(
    *,
    dataset_root: Path,
    output_root: Path,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    num_clients: int = 4,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create and audit all 38 PlantVillage classes for the main study."""

    return prepare_data_profiles(
        dataset_root=dataset_root,
        output_root=output_root,
        taxonomy=PLANTVILLAGE_FULL_TAXONOMY,
        profile_specs=FULL_PROFILE_SPECS,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        num_clients=num_clients,
        seed=seed,
    )


def prepare_data_profiles(
    *,
    dataset_root: Path,
    output_root: Path,
    taxonomy: DatasetTaxonomy,
    profile_specs: tuple[DataProfileSpec, ...],
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    num_clients: int = 4,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create and audit profiles with one shared, leakage-safe global test set.

    ``output_root`` must be empty so a prior experimental partition cannot be
    overwritten accidentally.  Every profile receives byte-identical master
    manifests and a different ``clients`` directory.

    ``dataset_root`` serves two purposes: it is scanned for images, and it is
    the anchor the manifests' relative paths are written against.  The profile
    set that comes out therefore describes the same experiment on any machine
    that holds the same dataset, wherever it is mounted.
    """

    output_root = _prepare_empty_output_root(output_root)
    resolved_dataset_root = dataset_root.expanduser().resolve()
    records = scan_plantvillage(resolved_dataset_root, taxonomy)
    train_records, test_records, split_statistics = (
        content_grouped_stratified_train_test_split(
            records,
            test_fraction=test_fraction,
            seed=seed,
            num_classes=len(taxonomy.class_names),
            dataset_root=resolved_dataset_root,
        )
    )

    profile_rows: list[dict[str, Any]] = []
    for spec in profile_specs:
        profile_rows.append(
            _build_profile(
                spec,
                output_root=output_root,
                taxonomy=taxonomy,
                train_records=train_records,
                test_records=test_records,
                num_train=len(train_records),
                num_test=len(test_records),
                test_fraction=test_fraction,
                validation_fraction=validation_fraction,
                num_clients=num_clients,
                seed=seed,
                dataset_root=resolved_dataset_root,
            )
        )

    test_hashes = {row["test_manifest_sha256"] for row in profile_rows}
    train_hashes = {row["train_manifest_sha256"] for row in profile_rows}
    all_audits_passed = all(row["audit_status"] == "passed" for row in profile_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "profile_set": f"{taxonomy.scope}_iid_non_iid",
        "taxonomy_scope": taxonomy.scope,
        "num_classes": len(taxonomy.class_names),
        "class_order": list(taxonomy.class_names),
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


def extend_data_profiles(
    *,
    output_root: Path,
    taxonomy: DatasetTaxonomy,
    profile_specs: tuple[DataProfileSpec, ...],
    dataset_root: Path,
    source_profile: str = "iid",
    validation_fraction: float = 0.2,
    num_clients: int = 4,
    seed: int = 2026,
) -> dict[str, Any]:
    """Add profiles to an existing set while preserving its global split.

    ``prepare_data_profiles`` refuses to write into a non-empty directory, which
    is what keeps a finished partition from being silently replaced.  Adding a
    skew profile to a set that already cost hours of partitioning therefore
    needs a separate, narrower door: this function never rescans the dataset and
    never recomputes a split.  It copies the existing train/test manifests
    byte-for-byte from ``source_profile`` and only repartitions the training
    records across clients, so D-024 — every profile shares one global test set
    — holds by construction rather than by convention.

    ``dataset_root`` is not scanned; it only anchors the manifest paths that are
    already recorded, so that partitioning can group images by content hash and
    the audit can open them.  It is required rather than optional: without it
    every image resolves to nothing, the audit reports each one as invalid, and
    the extension fails in a way that looks like corrupt data instead of a
    missing argument.

    Existing profile directories are never touched; a name that already exists
    is rejected instead of overwritten.
    """

    resolved_root = output_root.expanduser().resolve()
    resolved_dataset_root = dataset_root.expanduser().resolve()
    if not resolved_dataset_root.is_dir():
        raise NotADirectoryError(
            f"dataset root does not exist: {resolved_dataset_root}; the profiles "
            "store image paths relative to it and the audit could open none of them"
        )
    index_path = resolved_root / "profiles_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"no profile set to extend at {resolved_root}; run the prepare command first"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))

    if index.get("taxonomy_scope") != taxonomy.scope:
        raise ValueError(
            "refusing to mix taxonomies: existing set is "
            f"{index.get('taxonomy_scope')!r}, requested {taxonomy.scope!r}"
        )
    if int(index.get("num_clients", -1)) != num_clients:
        raise ValueError(
            f"existing set uses {index.get('num_clients')} clients, requested {num_clients}"
        )
    if int(index.get("seed", -1)) != seed:
        raise ValueError(
            f"existing set uses seed {index.get('seed')}, requested {seed}; "
            "a different seed would produce a different partition basis"
        )

    existing_names = [row["name"] for row in index["profiles"]]
    source_row = next(
        (row for row in index["profiles"] if row["name"] == source_profile), None
    )
    if source_row is None:
        raise ValueError(
            f"source profile {source_profile!r} not found in {existing_names}"
        )
    duplicates = [spec.name for spec in profile_specs if spec.name in existing_names]
    if duplicates:
        raise FileExistsError(f"refusing to overwrite existing profiles: {duplicates}")

    source_root = resolved_root / source_profile
    source_train = source_root / "train_manifest.csv"
    source_test = source_root / "test_manifest.csv"
    if _sha256_file(source_train) != source_row["train_manifest_sha256"]:
        raise ValueError(
            f"{source_train} no longer matches the checksum recorded in profiles_index.json"
        )
    if _sha256_file(source_test) != source_row["test_manifest_sha256"]:
        raise ValueError(
            f"{source_test} no longer matches the checksum recorded in profiles_index.json"
        )

    train_records = read_manifest(source_train)
    new_rows = [
        _build_profile(
            spec,
            output_root=resolved_root,
            taxonomy=taxonomy,
            train_records=train_records,
            source_train_manifest=source_train,
            source_test_manifest=source_test,
            num_train=int(source_row["num_train"]),
            num_test=int(source_row["num_test"]),
            test_fraction=float(source_row["test_fraction"]),
            validation_fraction=validation_fraction,
            num_clients=num_clients,
            seed=seed,
            dataset_root=resolved_dataset_root,
        )
        for spec in profile_specs
    ]

    profile_rows = [*index["profiles"], *new_rows]
    test_hashes = {row["test_manifest_sha256"] for row in profile_rows}
    train_hashes = {row["train_manifest_sha256"] for row in profile_rows}
    index["profiles"] = profile_rows
    index["status"] = (
        "passed"
        if all(row["audit_status"] == "passed" for row in profile_rows)
        else "failed"
    )
    index["shared_split_invariants"] = {
        "same_train_manifest": len(train_hashes) == 1,
        "same_global_test_manifest": len(test_hashes) == 1,
        "train_manifest_sha256": next(iter(train_hashes)),
        "test_manifest_sha256": next(iter(test_hashes)),
    }
    if not index["shared_split_invariants"]["same_global_test_manifest"]:
        raise AssertionError(
            "extension broke the shared global test set; refusing to record the result"
        )
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return index


def _build_profile(
    spec: DataProfileSpec,
    *,
    output_root: Path,
    taxonomy: DatasetTaxonomy,
    train_records: list,
    source_train_manifest: Path | None = None,
    source_test_manifest: Path | None = None,
    test_records: list | None = None,
    num_train: int,
    num_test: int,
    test_fraction: float,
    validation_fraction: float,
    num_clients: int,
    seed: int,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Write one profile directory and return its index row."""

    profile_root = output_root / spec.name
    train_manifest = profile_root / "train_manifest.csv"
    test_manifest = profile_root / "test_manifest.csv"
    client_root = profile_root / "clients"

    if source_train_manifest is not None and source_test_manifest is not None:
        # Copy rather than rewrite: a byte-identical manifest is the evidence
        # that the extension reused the original split instead of making one.
        profile_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_train_manifest, train_manifest)
        shutil.copyfile(source_test_manifest, test_manifest)
    else:
        assert test_records is not None
        write_manifest(train_records, train_manifest)
        write_manifest(test_records, test_manifest)

    write_client_manifests(
        train_records,
        client_root,
        num_clients=num_clients,
        partition_kind=spec.partition_kind,
        alpha=spec.dirichlet_alpha or 0.5,
        quantity_skew=spec.quantity_skew,
        feature_skew_strength=spec.feature_skew_strength,
        validation_fraction=validation_fraction,
        seed=seed,
        pooled_output_dir=profile_root,
        num_classes=len(taxonomy.class_names),
        dataset_root=dataset_root,
    )

    audit = audit_prepared_data(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        client_data_root=client_root,
        num_clients=num_clients,
        class_names=taxonomy.class_names,
        dataset_root=dataset_root,
    )
    audit_path = profile_root / "data_audit.json"
    write_audit_report(audit, audit_path)
    profile_metadata = {
        **asdict(spec),
        "taxonomy_scope": taxonomy.scope,
        "num_classes": len(taxonomy.class_names),
        "class_order": list(taxonomy.class_names),
        "num_clients": num_clients,
        "seed": seed,
        "num_train": num_train,
        "num_test": num_test,
        "test_fraction": test_fraction,
        "validation_fraction": validation_fraction,
        "train_manifest_sha256": _sha256_file(train_manifest),
        "pooled_train_manifest_sha256": _sha256_file(
            profile_root / "pooled_train_manifest.csv"
        ),
        "validation_manifest_sha256": _sha256_file(
            profile_root / "validation_manifest.csv"
        ),
        "test_manifest_sha256": _sha256_file(test_manifest),
        "partition_summary_sha256": _sha256_file(
            client_root / "partition_summary.json"
        ),
        "data_audit_sha256": _sha256_file(audit_path),
        "audit_status": audit["status"],
        "paths": {
            "train_manifest": str(train_manifest.relative_to(output_root)),
            "pooled_train_manifest": str(
                (profile_root / "pooled_train_manifest.csv").relative_to(output_root)
            ),
            "validation_manifest": str(
                (profile_root / "validation_manifest.csv").relative_to(output_root)
            ),
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
    return {**profile_metadata, "profile_sha256": _sha256_file(profile_path)}


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
