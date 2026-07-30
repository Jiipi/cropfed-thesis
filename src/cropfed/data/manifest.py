"""Build auditable CSV manifests from the PlantVillage tomato subset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cropfed.constants import PLANTVILLAGE_FOLDER_TO_CLASS, TOMATO_CLASSES
from cropfed.data.partitioning import make_partitions, partition_statistics

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    path: str
    label_id: int
    label_name: str
    split: str = "unassigned"


def scan_plantvillage_tomato(root: Path) -> list[ImageRecord]:
    """Scan expected PlantVillage folders without copying image bytes."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    records: list[ImageRecord] = []
    for folder_name, class_name in PLANTVILLAGE_FOLDER_TO_CLASS.items():
        class_dir = root / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing required class folder: {class_dir}")
        label_id = TOMATO_CLASSES.index(class_name)
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


def stratified_train_test_split(
    records: Iterable[ImageRecord],
    test_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split each class before client partitioning to prevent test leakage."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rows = list(records)
    rng = np.random.default_rng(seed)
    train: list[ImageRecord] = []
    test: list[ImageRecord] = []

    for label_id in range(len(TOMATO_CLASSES)):
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
    for label_id in range(len(TOMATO_CLASSES)):
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
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> list[dict[str, object]]:
    """Partition training records and create local train/validation manifests."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rows = list(train_records)
    labels = np.asarray([row.label_id for row in rows], dtype=np.int64)
    partitions = make_partitions(
        labels,
        num_clients,
        partition_kind,
        alpha=alpha,
        seed=seed,
        min_size=2,
    )
    rng = np.random.default_rng(seed + 17)
    summary = partition_statistics(labels, partitions, num_classes=len(TOMATO_CLASSES))

    for client_id, indices in enumerate(partitions):
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        validation_size = max(1, int(round(shuffled.size * validation_fraction)))
        if validation_size >= shuffled.size:
            validation_size = shuffled.size - 1
        validation_indices = set(shuffled[:validation_size].tolist())
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

        client_dir = destination_root / f"client_{client_id}"
        write_manifest(local_train, client_dir / "train_manifest.csv")
        write_manifest(local_validation, client_dir / "val_manifest.csv")
        summary[client_id]["num_train"] = len(local_train)
        summary[client_id]["num_validation"] = len(local_validation)

    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "partition_summary.json").write_text(
        json.dumps(
            {
                "partition_kind": partition_kind,
                "dirichlet_alpha": alpha if partition_kind == "dirichlet" else None,
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
