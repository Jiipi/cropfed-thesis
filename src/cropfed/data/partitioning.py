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
    quantity_skew: bool = False,
    feature_skew_strength: float = 0.5,
) -> list[IndexArray]:
    # ``quantity_skew`` is the profile-level flag; ``kind="quantity_skew"`` is the
    # equivalent request-level name used by the API, CLI and config boundary.
    if kind == "quantity_skew":
        kind, quantity_skew = "iid", True
    if kind == "feature_skew":
        if quantity_skew:
            raise ValueError("quantity_skew cannot be combined with feature_skew")
        return feature_skew_partition(
            labels, num_clients, feature_skew_strength, seed, min_size=min_size
        )
    if quantity_skew and kind != "iid":
        raise ValueError(
            "quantity_skew cannot be combined with Dirichlet label skew; "
            "use an explicit quantity-skew-only profile"
        )
    if kind == "iid":
        partitions = iid_partition(labels, num_clients, seed)
    elif kind == "dirichlet":
        partitions = dirichlet_partition(
            labels, num_clients, alpha, seed, min_size=min_size
        )
    else:
        raise ValueError(
            "kind must be 'iid', 'dirichlet', 'quantity_skew', or 'feature_skew'"
        )
    if quantity_skew:
        partitions = _apply_quantity_skew(partitions, seed, min_size=min_size)
    return partitions


def _apply_quantity_skew(
    partitions: list[IndexArray], seed: int, min_size: int = 1
) -> list[IndexArray]:
    """Redistribute every sample into deliberately unequal client shares.

    Quantity skew (lệch số lượng) varies how *much* data each site holds while
    leaving label proportions alone.  No index is dropped or duplicated.

    ``min_size`` is a floor on each client's share.  The caller partitions
    content groups rather than raw images, and a client left with a single
    group cannot be split into local train and validation, so the floor is a
    correctness requirement downstream rather than a cosmetic one.
    """

    if not partitions:
        return partitions
    all_indices = np.concatenate(partitions)
    num_clients = len(partitions)
    total = all_indices.size
    if total == 0 or num_clients == 0:
        return partitions
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if min_size * num_clients > total:
        raise ValueError("min_size is impossible for the given sample count")

    rng = np.random.default_rng(seed + 999)
    shuffled = rng.permutation(all_indices)

    proportions = rng.dirichlet(np.full(num_clients, 0.5))
    proportions = np.maximum(proportions, 0.05)
    proportions /= proportions.sum()

    # Allocate only the slack above the floor, so the floor cannot be undone by
    # the rounding-repair loop below.
    slack = total - min_size * num_clients
    sizes = min_size + np.floor(proportions * slack).astype(int)

    diff = total - int(sizes.sum())
    for offset in range(max(0, diff)):
        sizes[offset % num_clients] += 1
    for _ in range(max(0, -diff)):
        largest = int(np.argmax(sizes))
        if sizes[largest] <= min_size:
            break
        sizes[largest] -= 1

    result: list[IndexArray] = []
    offset = 0
    for size in sizes:
        result.append(shuffled[offset : offset + size])
        offset += size
    _assert_complete_partition(result, total)
    return result


def feature_skew_partition(
    labels: Sequence[int],
    num_clients: int,
    strength: float = 0.5,
    seed: int = 2026,
    min_size: int = 1,
) -> list[IndexArray]:
    """Partition each class into disjoint, unevenly sized per-client instance blocks.

    Scope, stated precisely so the thesis does not overclaim: this is a
    *sampling-level* approximation of feature skew (lệch đặc trưng).  Each client
    receives a different, non-overlapping subset of instances **within** every
    class, so per-client feature distributions differ through which photographs
    each site holds.  It does **not** modify pixels, so it does not reproduce a
    true covariate shift such as a different camera, illumination or background;
    that would require a per-client image transform in the data loader.

    ``strength`` controls how uneven the per-class blocks are: 0.0 gives nearly
    equal blocks (close to IID), 1.0 gives strongly unequal ones.  Label
    proportions stay approximately balanced by construction — that is what
    separates this from Dirichlet label skew.

    ``min_size`` is a floor on each client's share.  Uneven per-class blocks can
    otherwise leave a client with a single content group, which cannot be split
    into local train and validation downstream.
    """

    label_array = _validate_inputs(labels, num_clients)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("feature_skew_strength must be between 0 and 1")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if min_size * num_clients > label_array.size:
        raise ValueError("min_size is impossible for the given sample count")

    rng = np.random.default_rng(seed)
    classes = np.unique(label_array)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_position, class_id in enumerate(classes):
        class_indices = np.flatnonzero(label_array == class_id)
        rng.shuffle(class_indices)

        # A class may contain fewer samples than clients in small fixtures or
        # rare-class datasets. Rotate those samples across clients so no fixed
        # client is starved for every rare class.
        if len(class_indices) < num_clients:
            splits = [np.asarray([], dtype=np.int64) for _ in range(num_clients)]
            for offset, sample_index in enumerate(class_indices):
                client_id = (class_position + offset) % num_clients
                splits[client_id] = np.asarray([sample_index], dtype=np.int64)
        # Split the class samples into num_clients consecutive blocks.
        # Higher strength = more extreme block sizes.
        elif strength <= 0.01:
            # Near-IID: roughly equal splits
            splits = np.array_split(class_indices, num_clients)
        else:
            # Create uneven splits proportional to strength
            remaining = len(class_indices)
            splits = []
            for client_id in range(num_clients):
                if client_id == num_clients - 1:
                    size = remaining
                else:
                    # Base allocation + strength-weighted variation
                    base = remaining / (num_clients - client_id)
                    variation = base * strength * rng.uniform(-0.5, 0.5)
                    size = max(1, int(base + variation))
                    size = min(size, remaining - (num_clients - client_id - 1))
                start = len(class_indices) - remaining
                splits.append(class_indices[start : start + size])
                remaining -= size

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    partitions = []
    for indices in client_indices:
        rng.shuffle(indices)
        partitions.append(np.asarray(indices, dtype=np.int64))

    partitions = _enforce_min_size(partitions, min_size, rng)
    _assert_complete_partition(partitions, label_array.size)
    return partitions


def _enforce_min_size(
    partitions: list[IndexArray], min_size: int, rng: np.random.Generator
) -> list[IndexArray]:
    """Move samples from the largest clients until every client clears ``min_size``.

    Only the deficit is moved, so the intended skew is preserved as far as the
    floor allows.  Donors are chosen largest-first and are never taken below the
    floor themselves.
    """

    if min_size <= 1:
        return partitions
    working = [list(part.tolist()) for part in partitions]
    for client_id, indices in enumerate(working):
        while len(indices) < min_size:
            donor = max(
                range(len(working)),
                key=lambda other: len(working[other]) if other != client_id else -1,
            )
            if len(working[donor]) <= min_size:
                raise ValueError(
                    "cannot satisfy min_size without starving another client; "
                    "reduce num_clients or min_size"
                )
            indices.append(working[donor].pop())
    result = []
    for indices in working:
        rng.shuffle(indices)
        result.append(np.asarray(indices, dtype=np.int64))
    return result


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
