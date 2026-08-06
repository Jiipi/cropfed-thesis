"""Integrity audit for prepared image manifests and client partitions.

The audit runs locally where the manifests and images live. Its JSON report
contains hashes, IDs, counts, and issue codes, but never image bytes or local
image paths, so it can be retained with experiment metadata safely.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from cropfed.data.manifest import IMAGE_EXTENSIONS, ImageRecord, read_manifest


def audit_prepared_data(
    *,
    train_manifest: Path,
    test_manifest: Path,
    client_data_root: Path | None = None,
    num_clients: int = 4,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Audit image integrity, taxonomy, split isolation, and client assignment.

    Content hashes are computed once per unique local path even though a sample
    normally appears in both the master training manifest and one client
    manifest. Cross-split content overlap is an error; exact duplicates wholly
    within train or test are reported as warnings for an explicit data decision.
    """

    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    resolved_class_names = tuple(class_names)
    if len(resolved_class_names) < 2 or len(set(resolved_class_names)) != len(
        resolved_class_names
    ):
        raise ValueError("class_names must contain at least two unique names")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    specifications: dict[str, tuple[Path, str]] = {
        "master_train": (train_manifest, "train"),
        "global_test": (test_manifest, "test"),
    }
    if client_data_root is not None:
        for client_id in range(num_clients):
            client_dir = client_data_root / f"client_{client_id}"
            specifications[f"client_{client_id}_train"] = (
                client_dir / "train_manifest.csv",
                "local_train",
            )
            specifications[f"client_{client_id}_val"] = (
                client_dir / "val_manifest.csv",
                "local_val",
            )

    records_by_scope: dict[str, list[ImageRecord]] = {}
    manifest_summaries: dict[str, dict[str, Any]] = {}
    for scope, (manifest_path, expected_split) in specifications.items():
        if not manifest_path.is_file():
            _add_issue(errors, "manifest_missing", scope=scope)
            records_by_scope[scope] = []
            continue
        try:
            records = read_manifest(manifest_path)
        except Exception as error:
            _add_issue(
                errors,
                "manifest_unreadable",
                scope=scope,
                reason=type(error).__name__,
            )
            records_by_scope[scope] = []
            continue

        records_by_scope[scope] = records
        split_counts = Counter(record.split for record in records)
        class_counts = [0] * len(resolved_class_names)
        for record in records:
            if 0 <= record.label_id < len(resolved_class_names):
                class_counts[record.label_id] += 1
        manifest_summaries[scope] = {
            "sha256": _sha256_file(manifest_path),
            "num_records": len(records),
            "split_counts": dict(sorted(split_counts.items())),
            "class_counts": class_counts,
        }

        if not records:
            _add_issue(errors, "manifest_empty", scope=scope)
        unexpected_splits = sorted(set(split_counts) - {expected_split})
        if unexpected_splits:
            _add_issue(
                errors,
                "unexpected_split_value",
                scope=scope,
                expected=expected_split,
                actual=unexpected_splits,
            )

        duplicate_ids = _duplicates(record.image_id for record in records)
        if duplicate_ids:
            _add_issue(
                errors,
                "duplicate_image_id_within_manifest",
                scope=scope,
                count=len(duplicate_ids),
                image_ids=duplicate_ids,
            )
        duplicate_paths = _duplicates(_path_key(record.path) for record in records)
        if duplicate_paths:
            _add_issue(
                errors,
                "duplicate_path_within_manifest",
                scope=scope,
                count=len(duplicate_paths),
            )

        invalid_taxonomy: list[str] = []
        for record in records:
            if not 0 <= record.label_id < len(resolved_class_names):
                invalid_taxonomy.append(record.image_id)
            elif record.label_name != resolved_class_names[record.label_id]:
                invalid_taxonomy.append(record.image_id)
        if invalid_taxonomy:
            _add_issue(
                errors,
                "taxonomy_mismatch",
                scope=scope,
                count=len(invalid_taxonomy),
                image_ids=sorted(set(invalid_taxonomy)),
            )

    for scope in ("master_train", "global_test"):
        present = {
            record.label_id
            for record in records_by_scope.get(scope, [])
            if 0 <= record.label_id < len(resolved_class_names)
        }
        missing_classes = sorted(set(range(len(resolved_class_names))) - present)
        if missing_classes:
            _add_issue(
                errors,
                "missing_taxonomy_classes",
                scope=scope,
                class_ids=missing_classes,
            )

    path_references: dict[str, set[str]] = defaultdict(set)
    for records in records_by_scope.values():
        for record in records:
            path_references[_path_key(record.path)].add(record.image_id)

    image_inspections: dict[str, dict[str, Any]] = {}
    invalid_images: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    for path_key, image_ids in sorted(path_references.items()):
        image_path = Path(path_key)
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            issue = {
                "image_ids": sorted(image_ids),
                "reason": "unsupported_extension",
            }
            invalid_images.append(issue)
            _add_issue(errors, "invalid_image", **issue)
            continue
        inspection = _inspect_image(image_path)
        if "error" in inspection:
            issue = {
                "image_ids": sorted(image_ids),
                "reason": inspection["error"],
            }
            invalid_images.append(issue)
            _add_issue(errors, "invalid_image", **issue)
            continue
        image_inspections[path_key] = inspection
        format_counts[str(inspection["format"])] += 1

    train_records = records_by_scope.get("master_train", [])
    test_records = records_by_scope.get("global_test", [])
    train_ids = {record.image_id for record in train_records}
    test_ids = {record.image_id for record in test_records}
    train_paths = {_path_key(record.path) for record in train_records}
    test_paths = {_path_key(record.path) for record in test_records}
    train_hashes = _content_hashes(train_records, image_inspections)
    test_hashes = _content_hashes(test_records, image_inspections)

    id_overlap = sorted(train_ids & test_ids)
    path_overlap_count = len(train_paths & test_paths)
    content_overlap = sorted(train_hashes & test_hashes)
    if id_overlap:
        _add_issue(
            errors,
            "cross_split_image_id_overlap",
            count=len(id_overlap),
            image_ids=id_overlap,
        )
    if path_overlap_count:
        _add_issue(
            errors,
            "cross_split_path_overlap",
            count=path_overlap_count,
        )
    if content_overlap:
        _add_issue(
            errors,
            "cross_split_content_overlap",
            count=len(content_overlap),
            sha256=content_overlap,
        )

    duplicate_groups = _duplicate_content_groups(
        train_records,
        test_records,
        image_inspections,
    )
    within_train = [group for group in duplicate_groups if group["splits"] == ["train"]]
    within_test = [group for group in duplicate_groups if group["splits"] == ["test"]]
    if within_train:
        _add_issue(
            warnings,
            "duplicate_content_within_train",
            count=len(within_train),
        )
    if within_test:
        _add_issue(
            warnings,
            "duplicate_content_within_test",
            count=len(within_test),
        )

    client_assignment = _audit_client_assignment(
        records_by_scope=records_by_scope,
        train_records=train_records,
        test_ids=test_ids,
        test_hashes=test_hashes,
        image_inspections=image_inspections,
        num_clients=num_clients,
        enabled=client_data_root is not None,
        errors=errors,
    )

    return {
        "report_kind": "prepared_data_integrity_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "failed" if errors else "passed",
        "taxonomy": {
            "num_classes": len(resolved_class_names),
            "class_order": list(resolved_class_names),
        },
        "manifests": manifest_summaries,
        "images": {
            "unique_paths_checked": len(path_references),
            "verified_images": len(image_inspections),
            "invalid_images": invalid_images,
            "format_counts": dict(sorted(format_counts.items())),
        },
        "duplicates": {
            "content_groups": duplicate_groups,
            "within_train_groups": len(within_train),
            "within_test_groups": len(within_test),
        },
        "global_split_overlap": {
            "image_id_count": len(id_overlap),
            "path_count": path_overlap_count,
            "content_sha256_count": len(content_overlap),
            "content_sha256": content_overlap,
        },
        "client_assignment": client_assignment,
        "errors": errors,
        "warnings": warnings,
        "privacy": {
            "contains_image_bytes": False,
            "contains_local_image_paths": False,
        },
    }


def write_audit_report(report: dict[str, Any], destination: Path) -> None:
    """Write a UTF-8 JSON audit artifact."""

    import json

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _audit_client_assignment(
    *,
    records_by_scope: dict[str, list[ImageRecord]],
    train_records: list[ImageRecord],
    test_ids: set[str],
    test_hashes: set[str],
    image_inspections: dict[str, dict[str, Any]],
    num_clients: int,
    enabled: bool,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    if not enabled:
        return {"checked": False}

    client_records: list[tuple[str, ImageRecord]] = []
    client_train_records: list[ImageRecord] = []
    client_validation_records: list[ImageRecord] = []
    client_counts: list[dict[str, int]] = []
    empty_client_splits = 0
    for client_id in range(num_clients):
        train_scope = f"client_{client_id}_train"
        val_scope = f"client_{client_id}_val"
        local_train = records_by_scope.get(train_scope, [])
        local_val = records_by_scope.get(val_scope, [])
        client_counts.append(
            {
                "client_id": client_id,
                "num_train": len(local_train),
                "num_validation": len(local_val),
            }
        )
        if not local_train:
            empty_client_splits += 1
            _add_issue(errors, "client_train_empty", client_id=client_id)
        if not local_val:
            empty_client_splits += 1
            _add_issue(errors, "client_validation_empty", client_id=client_id)
        client_records.extend((train_scope, record) for record in local_train)
        client_records.extend((val_scope, record) for record in local_val)
        client_train_records.extend(local_train)
        client_validation_records.extend(local_val)

    assignments: dict[str, list[str]] = defaultdict(list)
    for scope, record in client_records:
        assignments[record.image_id].append(scope)
    repeated_assignments = {
        image_id: sorted(scopes)
        for image_id, scopes in assignments.items()
        if len(scopes) > 1
    }
    if repeated_assignments:
        _add_issue(
            errors,
            "sample_assigned_to_multiple_client_splits",
            count=len(repeated_assignments),
            image_ids=sorted(repeated_assignments),
        )

    train_content_hashes = _content_hashes(client_train_records, image_inspections)
    validation_content_hashes = _content_hashes(
        client_validation_records, image_inspections
    )
    train_validation_content_overlap = sorted(
        train_content_hashes & validation_content_hashes
    )
    if train_validation_content_overlap:
        _add_issue(
            errors,
            "client_train_validation_content_overlap",
            count=len(train_validation_content_overlap),
            sha256=train_validation_content_overlap,
        )

    scopes_by_content: dict[str, set[str]] = defaultdict(set)
    for scope, record in client_records:
        inspection = image_inspections.get(_path_key(record.path))
        if inspection is not None:
            scopes_by_content[str(inspection["sha256"])].add(scope)
    multi_scope_content = {
        digest: sorted(scopes)
        for digest, scopes in scopes_by_content.items()
        if len(scopes) > 1
    }
    if multi_scope_content:
        _add_issue(
            errors,
            "duplicate_content_assigned_to_multiple_client_scopes",
            count=len(multi_scope_content),
            sha256=sorted(multi_scope_content),
        )

    train_by_id = {record.image_id: record for record in train_records}
    assigned_ids = set(assignments)
    expected_ids = set(train_by_id)
    missing_ids = sorted(expected_ids - assigned_ids)
    unexpected_ids = sorted(assigned_ids - expected_ids)
    if missing_ids:
        _add_issue(
            errors,
            "master_train_samples_missing_from_clients",
            count=len(missing_ids),
            image_ids=missing_ids,
        )
    if unexpected_ids:
        _add_issue(
            errors,
            "unexpected_client_samples",
            count=len(unexpected_ids),
            image_ids=unexpected_ids,
        )

    metadata_mismatches: set[str] = set()
    for _, record in client_records:
        source = train_by_id.get(record.image_id)
        if source is None:
            continue
        if (
            record.label_id != source.label_id
            or record.label_name != source.label_name
            or _path_key(record.path) != _path_key(source.path)
        ):
            metadata_mismatches.add(record.image_id)
    if metadata_mismatches:
        _add_issue(
            errors,
            "client_metadata_mismatch",
            count=len(metadata_mismatches),
            image_ids=sorted(metadata_mismatches),
        )

    client_test_id_overlap = sorted(assigned_ids & test_ids)
    client_hashes = _content_hashes(
        [record for _, record in client_records],
        image_inspections,
    )
    client_test_content_overlap = sorted(client_hashes & test_hashes)
    if client_test_id_overlap:
        _add_issue(
            errors,
            "client_global_test_image_id_overlap",
            count=len(client_test_id_overlap),
            image_ids=client_test_id_overlap,
        )
    if client_test_content_overlap:
        _add_issue(
            errors,
            "client_global_test_content_overlap",
            count=len(client_test_content_overlap),
            sha256=client_test_content_overlap,
        )

    return {
        "checked": True,
        "num_clients": num_clients,
        "clients": client_counts,
        "expected_master_train_samples": len(expected_ids),
        "assigned_unique_samples": len(assigned_ids),
        "missing_samples": len(missing_ids),
        "unexpected_samples": len(unexpected_ids),
        "multiply_assigned_samples": len(repeated_assignments),
        "metadata_mismatches": len(metadata_mismatches),
        "empty_client_splits": empty_client_splits,
        "global_test_image_id_overlap": len(client_test_id_overlap),
        "global_test_content_overlap": len(client_test_content_overlap),
        "train_validation_content_overlap": len(train_validation_content_overlap),
        "multi_scope_duplicate_content": len(multi_scope_content),
        "complete": not any(
            (
                missing_ids,
                unexpected_ids,
                repeated_assignments,
                metadata_mismatches,
                empty_client_splits,
                client_test_id_overlap,
                client_test_content_overlap,
                train_validation_content_overlap,
                multi_scope_content,
            )
        ),
    }


def _inspect_image(path: Path) -> dict[str, Any]:
    try:
        digest = _sha256_file(path)
        with Image.open(path) as source:
            image_format = source.format or "unknown"
            width, height = source.size
            source.verify()
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            rgb.load()
        return {
            "sha256": digest,
            "format": image_format,
            "width": width,
            "height": height,
        }
    except (OSError, ValueError, SyntaxError) as error:
        return {"error": type(error).__name__}


def _duplicate_content_groups(
    train_records: list[ImageRecord],
    test_records: list[ImageRecord],
    inspections: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_hash: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for split, records in (("train", train_records), ("test", test_records)):
        for record in records:
            inspection = inspections.get(_path_key(record.path))
            if inspection is not None:
                by_hash[str(inspection["sha256"])].add((split, record.image_id))
    groups: list[dict[str, Any]] = []
    for digest, entries in sorted(by_hash.items()):
        image_ids = sorted({image_id for _, image_id in entries})
        if len(image_ids) > 1:
            groups.append(
                {
                    "sha256": digest,
                    "image_ids": image_ids,
                    "splits": sorted({split for split, _ in entries}),
                }
            )
    return groups


def _content_hashes(
    records: list[ImageRecord], inspections: dict[str, dict[str, Any]]
) -> set[str]:
    return {
        str(inspections[_path_key(record.path)]["sha256"])
        for record in records
        if _path_key(record.path) in inspections
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_key(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _duplicates(values) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _add_issue(target: list[dict[str, Any]], code: str, **details: Any) -> None:
    target.append({"code": code, **details})
