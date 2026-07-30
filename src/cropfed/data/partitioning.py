"""Deterministic IID and label-skew Non-IID partitioning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

IndexArray = NDArray[np.int64]


def _validate_inputs(labels: Sequence[int], num_clients: int) -> NDArray[np.int64]:
    array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    if num_clients > array.size:
        raise ValueError("num_clients cannot exceed number of samples")
    if np.any(array < 0):
        raise ValueError("labels must be non-negative integers")
    return array


def iid_partition(
    labels: Sequence[int], num_clients: int, seed: int = 2026
) -> list[IndexArray]:
    """Randomly split all samples into near-equal client partitions."""

    label_array = _validate_inputs(labels, num_clients)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(label_array.size)
    return [part.astype(np.int64, copy=False) for part in np.array_split(shuffled, num_clients)]


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 2026,
    min_size: int = 1,
    max_retries: int = 1_000,
) -> list[IndexArray]:
    """Create label-skew partitions with a symmetric Dirichlet distribution.

    A smaller ``alpha`` generally yields more heterogeneous label proportions.
    The function retries because an unconstrained Dirichlet draw can leave a
    client empty. Every source index is assigned exactly once.
    """

    label_array = _validate_inputs(labels, num_clients)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if min_size * num_clients > label_array.size:
        raise ValueError("min_size is impossible for the given sample count")

    rng = np.random.default_rng(seed)
    classes = np.unique(label_array)

    for _ in range(max_retries):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for class_id in classes:
            class_indices = np.flatnonzero(label_array == class_id)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(class_indices.size, proportions)
            offset = 0
            for client_id, count in enumerate(counts):
                next_offset = offset + int(count)
                client_indices[client_id].extend(class_indices[offset:next_offset].tolist())
                offset = next_offset

        if min(map(len, client_indices)) >= min_size:
            partitions = []
            for indices in client_indices:
                rng.shuffle(indices)
                partitions.append(np.asarray(indices, dtype=np.int64))
            _assert_complete_partition(partitions, label_array.size)
            return partitions

    raise RuntimeError(
        "could not create a Dirichlet partition satisfying min_size; "
        "increase alpha, reduce num_clients, or reduce min_size"
    )


def make_partitions(
    labels: Sequence[int],
    num_clients: int,
    kind: str,
    alpha: float = 0.5,
    seed: int = 2026,
    min_size: int = 1,
) -> list[IndexArray]:
    if kind == "iid":
        return iid_partition(labels, num_clients, seed)
    if kind == "dirichlet":
        return dirichlet_partition(
            labels, num_clients, alpha, seed, min_size=min_size
        )
    raise ValueError("kind must be 'iid' or 'dirichlet'")


def partition_statistics(
    labels: Sequence[int],
    partitions: Sequence[Sequence[int]],
    num_classes: int | None = None,
) -> list[dict[str, object]]:
    """Return sample counts and class proportions for each client."""

    label_array = np.asarray(labels, dtype=np.int64)
    inferred_classes = int(label_array.max()) + 1
    class_count = inferred_classes if num_classes is None else num_classes
    if class_count < inferred_classes:
        raise ValueError("num_classes is smaller than the largest label")

    result: list[dict[str, object]] = []
    for client_id, indices in enumerate(partitions):
        index_array = np.asarray(indices, dtype=np.int64)
        counts = np.bincount(label_array[index_array], minlength=class_count)
        total = int(index_array.size)
        proportions = counts / total if total else np.zeros(class_count)
        result.append(
            {
                "client_id": client_id,
                "num_samples": total,
                "class_counts": counts.astype(int).tolist(),
                "class_proportions": proportions.astype(float).tolist(),
            }
        )
    return result


def _assert_complete_partition(partitions: Sequence[IndexArray], num_samples: int) -> None:
    combined = np.concatenate(partitions)
    if combined.size != num_samples:
        raise AssertionError("partition lost or duplicated samples")
    if not np.array_equal(np.sort(combined), np.arange(num_samples, dtype=np.int64)):
        raise AssertionError("partition indices are not a permutation of source indices")
