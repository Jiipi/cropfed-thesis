"""FedAvg-compatible weighted aggregation for tests and smoke simulations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Weights = dict[str, NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class ClientUpdate:
    client_id: str
    weights: Weights
    num_examples: int
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_examples <= 0:
            raise ValueError("num_examples must be positive")
        if not self.weights:
            raise ValueError("weights cannot be empty")

    @property
    def num_bytes(self) -> int:
        return int(sum(value.nbytes for value in self.weights.values()))


def weighted_average_updates(updates: list[ClientUpdate]) -> Weights:
    """Aggregate model parameters using each client's sample count."""

    _validate_updates(updates)
    reference = updates[0].weights
    total_examples = sum(update.num_examples for update in updates)
    result: Weights = {}

    for key, reference_value in reference.items():
        accumulator = np.zeros(reference_value.shape, dtype=np.float64)
        for update in updates:
            accumulator += update.weights[key].astype(np.float64) * update.num_examples
        averaged = accumulator / total_examples
        if np.issubdtype(reference_value.dtype, np.integer):
            averaged = np.rint(averaged)
        result[key] = averaged.astype(reference_value.dtype)
    return result


def weighted_average_metrics(updates: list[ClientUpdate]) -> dict[str, float]:
    """Aggregate only metrics present on every client."""

    _validate_updates(updates)
    common_keys = set(updates[0].metrics)
    for update in updates[1:]:
        common_keys.intersection_update(update.metrics)
    total_examples = sum(update.num_examples for update in updates)
    return {
        key: float(
            sum(update.metrics[key] * update.num_examples for update in updates)
            / total_examples
        )
        for key in sorted(common_keys)
    }


def _validate_updates(updates: list[ClientUpdate]) -> None:
    if not updates:
        raise ValueError("at least one client update is required")
    reference_keys = set(updates[0].weights)
    reference_shapes = {key: value.shape for key, value in updates[0].weights.items()}
    for update in updates:
        if set(update.weights) != reference_keys:
            raise ValueError("all clients must provide identical parameter keys")
        for key, value in update.weights.items():
            if value.shape != reference_shapes[key]:
                raise ValueError(f"shape mismatch for parameter {key!r}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite values in parameter {key!r}")
