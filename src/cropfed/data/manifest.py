"""Build auditable CSV manifests from PlantVillage folders."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cropfed.constants import (
    TOMATO_TAXONOMY,
    DatasetTaxonomy,
)
from cropfed.data.partitioning import make_partitions, partition_statistics

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    path: str
    label_id: int
    label_name: str
    split: str = "unassigned"


def scan_plantvillage(
    root: Path,
    taxonomy: DatasetTaxonomy,
) -> list[ImageRecord]:
    """Scan every folder required by ``taxonomy`` without copying image bytes."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    records: list[ImageRecord] = []
    for folder_name, class_name in taxonomy.folder_to_class.items():
        class_dir = root / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing required class folder: {class_dir}")
        label_id = taxonomy.class_names.index(class_name)
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                relative = path.relative_to(root).as_posix()
                image_id = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
                records.append(
                    ImageRecord(
                        image_id=image_id,
                        path=str(path),
                        label_id=label_id,
                        label_name=class_name,
                    )
                )
    if not records:
        raise ValueError("no supported image files found")
    return records


def scan_plantvillage_tomato(root: Path) -> list[ImageRecord]:
    """Compatibility wrapper for the original ten-class tomato pilot."""

    return scan_plantvillage(root, TOMATO_TAXONOMY)


def stratified_train_test_split(
    records: Iterable[ImageRecord],
    test_fraction: float = 0.2,
    seed: int = 2026,
    *,
    num_classes: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split each class before client partitioning to prevent test leakage."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rows = list(records)
    rng = np.random.default_rng(seed)
    train: list[ImageRecord] = []
    test: list[ImageRecord] = []

    for label_id in range(num_classes):
        class_rows = [row for row in rows if row.label_id == label_id]
        if len(class_rows) < 2:
            raise ValueError(f"class {label_id} needs at least two images")
        order = rng.permutation(len(class_rows))
        test_size = max(1, int(round(len(class_rows) * test_fraction)))
        test_indices = set(order[:test_size].tolist())
        for index, row in enumerate(class_rows):
            split = "test" if index in test_indices else "train"
            target = test if split == "test" else train
            target.append(
                ImageRecord(
                    image_id=row.image_id,
                    path=row.path,
                    label_id=row.label_id,
                    label_name=row.label_name,
                    split=split,
                )
            )
    return train, test


def content_grouped_stratified_train_test_split(
    records: Iterable[ImageRecord],
    test_fraction: float = 0.2,
    seed: int = 2026,
    *,
    num_classes: int,
) -> tuple[list[ImageRecord], list[ImageRecord], dict[str, object]]:
    """Stratify without allowing exact-content duplicates across the split.

    PlantVillage contains a small number of byte-identical files under distinct
    paths.  Treating paths independently can leak the same pixels into train and
    test.  This splitter hashes each source image once and assigns every identical
    content group atomically.  A hash shared by different labels is rejected.
    """

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rows = list(records)
    if not rows:
        raise ValueError("records must not be empty")

    groups_by_hash: dict[str, list[int]] = defaultdict(list)
    labels_by_hash: dict[str, set[int]] = defaultdict(set)
    hashes_by_index: list[str] = []
    for index, row in enumerate(rows):
        digest = _sha256_file(Path(row.path))
        hashes_by_index.append(digest)
        groups_by_hash[digest].append(index)
        labels_by_hash[digest].add(row.label_id)
    conflicting = [digest for digest, labels in labels_by_hash.items() if len(labels) > 1]
    if conflicting:
        raise ValueError(
            "identical image content appears under different labels: "
            f"groups={len(conflicting)}"
        )

    rng = np.random.default_rng(seed)
    test_hashes: set[str] = set()
    class_summary: list[dict[str, int]] = []
    for label_id in range(num_classes):
        class_hashes = [
            digest
            for digest, labels in labels_by_hash.items()
            if labels == {label_id}
        ]
        if len(class_hashes) < 2:
            raise ValueError(f"class {label_id} needs at least two content groups")
        order = rng.permutation(len(class_hashes))
        shuffled = [class_hashes[int(index)] for index in order]
        num_class_images = sum(len(groups_by_hash[digest]) for digest in shuffled)
        target_test = max(1, int(round(num_class_images * test_fraction)))
        selected: list[str] = []
        selected_images = 0

        # The final group is reserved for train. Add a group only while it moves
        # the image count closer to the requested fraction. Most groups are size 1.
        for digest in shuffled[:-1]:
            candidate = selected_images + len(groups_by_hash[digest])
            if not selected or abs(candidate - target_test) < abs(
                selected_images - target_test
            ):
                selected.append(digest)
                selected_images = candidate
            if selected_images == target_test:
                break
        test_hashes.update(selected)
        class_summary.append(
            {
                "label_id": label_id,
                "num_images": num_class_images,
                "num_content_groups": len(class_hashes),
                "num_test_images": selected_images,
                "num_train_images": num_class_images - selected_images,
            }
        )

    train: list[ImageRecord] = []
    test: list[ImageRecord] = []
    for row, digest in zip(rows, hashes_by_index, strict=True):
        split = "test" if digest in test_hashes else "train"
        target = test if split == "test" else train
        target.append(
            ImageRecord(
                image_id=row.image_id,
                path=row.path,
                label_id=row.label_id,
                label_name=row.label_name,
                split=split,
            )
        )

    duplicate_groups = [indices for indices in groups_by_hash.values() if len(indices) > 1]
    statistics: dict[str, object] = {
        "strategy": "sha256_content_grouped_stratified",
        "num_images": len(rows),
        "num_unique_content_groups": len(groups_by_hash),
        "num_duplicate_content_groups": len(duplicate_groups),
        "num_redundant_images": sum(len(group) - 1 for group in duplicate_groups),
        "cross_label_conflict_groups": 0,
        "classes": class_summary,
    }
    return train, test, statistics


def write_manifest(records: Iterable[ImageRecord], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "path", "label_id", "label_name", "split"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "image_id": row.image_id,
                    "path": row.path,
                    "label_id": row.label_id,
                    "label_name": row.label_name,
                    "split": row.split,
                }
            )


def write_client_manifests(
    train_records: Iterable[ImageRecord],
    destination_root: Path,
    *,
    num_clients: int = 4,
    partition_kind: str = "dirichlet",
    alpha: float = 0.5,
    quantity_skew: bool = False,
    feature_skew_strength: float = 0.5,
    validation_fraction: float = 0.2,
    seed: int = 2026,
    pooled_output_dir: Path | None = None,
    num_classes: int,
) -> list[dict[str, object]]:
    """Partition records and create local plus optional pooled train/validation files."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rows = list(train_records)
    labels = np.asarray([row.label_id for row in rows], dtype=np.int64)
    content_keys = [_client_partition_group_key(row) for row in rows]
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for index, (row, content_key) in enumerate(
        zip(rows, content_keys, strict=True)
    ):
        existing_label = grouped_labels.setdefault(content_key, row.label_id)
        if existing_label != row.label_id:
            raise ValueError("identical client-partition content has conflicting labels")
        grouped_indices[content_key].append(index)
    content_groups = list(grouped_indices.values())
    group_labels = np.asarray(
        [rows[indices[0]].label_id for indices in content_groups],
        dtype=np.int64,
    )
    grouped_partitions = make_partitions(
        group_labels,
        num_clients,
        partition_kind,
        alpha=alpha,
        seed=seed,
        min_size=2,
        quantity_skew=quantity_skew,
        feature_skew_strength=feature_skew_strength,
    )
    partitions = [
        np.asarray(
            [
                source_index
                for group_index in group_partition
                for source_index in content_groups[int(group_index)]
            ],
            dtype=np.int64,
        )
        for group_partition in grouped_partitions
    ]
    rng = np.random.default_rng(seed + 17)
    summary = partition_statistics(labels, partitions, num_classes=num_classes)
    pooled_train: list[ImageRecord] = []
    pooled_validation: list[ImageRecord] = []

    for client_id, indices in enumerate(partitions):
        local_groups: dict[str, list[int]] = defaultdict(list)
        for source_index in indices:
            local_groups[content_keys[int(source_index)]].append(int(source_index))
        shuffled_groups = list(local_groups.values())
        rng.shuffle(shuffled_groups)
        validation_group_count = max(
            1,
            int(round(len(shuffled_groups) * validation_fraction)),
        )
        if validation_group_count >= len(shuffled_groups):
            validation_group_count = len(shuffled_groups) - 1
        if validation_group_count < 1:
            raise ValueError(
                f"client {client_id} needs at least two content groups for train/validation"
            )
        validation_indices = {
            source_index
            for group in shuffled_groups[:validation_group_count]
            for source_index in group
        }
        local_train: list[ImageRecord] = []
        local_validation: list[ImageRecord] = []
        for source_index in indices:
            source = rows[int(source_index)]
            is_validation = int(source_index) in validation_indices
            split = "local_val" if is_validation else "local_train"
            record = ImageRecord(
                image_id=source.image_id,
                path=source.path,
                label_id=source.label_id,
                label_name=source.label_name,
                split=split,
            )
            (local_validation if is_validation else local_train).append(record)

        pooled_train.extend(
            ImageRecord(
                image_id=record.image_id,
                path=record.path,
                label_id=record.label_id,
                label_name=record.label_name,
                split="train",
            )
            for record in local_train
        )
        pooled_validation.extend(
            ImageRecord(
                image_id=record.image_id,
                path=record.path,
                label_id=record.label_id,
                label_name=record.label_name,
                split="validation",
            )
            for record in local_validation
        )

        client_dir = destination_root / f"client_{client_id}"
        write_manifest(local_train, client_dir / "train_manifest.csv")
        write_manifest(local_validation, client_dir / "val_manifest.csv")
        summary[client_id]["num_train"] = len(local_train)
        summary[client_id]["num_validation"] = len(local_validation)

    if pooled_output_dir is not None:
        write_manifest(
            sorted(pooled_train, key=lambda record: record.image_id),
            pooled_output_dir / "pooled_train_manifest.csv",
        )
        write_manifest(
            sorted(pooled_validation, key=lambda record: record.image_id),
            pooled_output_dir / "validation_manifest.csv",
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "partition_summary.json").write_text(
        json.dumps(
            {
                "partition_kind": partition_kind,
                "skew_type": (
                    "quantity"
                    if quantity_skew
                    else "feature"
                    if partition_kind == "feature_skew"
                    else "label"
                    if partition_kind == "dirichlet"
                    else "none"
                ),
                "dirichlet_alpha": alpha if partition_kind == "dirichlet" else None,
                "quantity_skew": quantity_skew,
                "feature_skew_strength": (
                    feature_skew_strength
                    if partition_kind == "feature_skew"
                    else None
                ),
                "num_clients": num_clients,
                "seed": seed,
                "clients": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def read_manifest(path: Path) -> list[ImageRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "path", "label_id", "label_name", "split"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"manifest columns must be exactly {sorted(required)}")
        return [
            ImageRecord(
                image_id=row["image_id"],
                path=row["path"],
                label_id=int(row["label_id"]),
                label_name=row["label_name"],
                split=row["split"],
            )
            for row in reader
        ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _client_partition_group_key(record: ImageRecord) -> str:
    path = Path(record.path).expanduser()
    if path.is_file():
        return f"sha256:{_sha256_file(path)}"
    # Dependency-light unit tests may use virtual paths. Unique IDs retain the
    # historical behavior without weakening production grouping of real files.
    return f"image-id:{record.image_id}"
