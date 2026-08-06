"""FedAvg-compatible weighted aggregation for tests and smoke simulations.

Implements the aggregation side of:
- FedAvg (weighted average)
- FedProx (same server aggregation, proximal term on client)
- FedBN  (exclude batch-norm parameters from aggregation)
- SCAFFOLD (server maintains control variate)
- MOON (model-contrastive loss on client, standard aggregation on server)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Weights = dict[str, NDArray[np.generic]]

# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FedAvg – weighted sample-size average
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FedBN – exclude batch-norm layers from aggregation
# ---------------------------------------------------------------------------


_BN_SPECIFIC_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def _identify_batch_norm_parameter_prefixes(keys: set[str]) -> set[str]:
    """Return the set of prefixes that belong to BatchNorm layers.

    BatchNorm parameters are uniquely identified by ``.running_mean``,
    ``.running_var``, and ``.num_batches_tracked`` — no other layer type
    has these.  Any ``.weight`` or ``.bias`` sharing the same prefix is
    also a BN parameter.
    """
    bn_prefixes: set[str] = set()
    for key in keys:
        for suffix in _BN_SPECIFIC_SUFFIXES:
            if key.endswith(suffix):
                # The prefix is everything before the suffix
                prefix = key[: -len(suffix)]
                bn_prefixes.add(prefix)
                break
    return bn_prefixes


def batch_norm_parameter_names(keys: Iterable[str]) -> set[str]:
    """Return the subset of *keys* that belong to BatchNorm layers.

    Shared by the smoke simulator's aggregation and the real Flower client so
    both agree on exactly which tensors FedBN keeps local.  A disagreement here
    would make ``fedbn`` mean two different things in the same thesis.
    """

    key_set = set(keys)
    bn_prefixes = _identify_batch_norm_parameter_prefixes(key_set)
    return {name for name in key_set if _is_batch_norm_parameter(name, bn_prefixes)}


def _is_batch_norm_parameter(name: str, bn_prefixes: set[str]) -> bool:
    """Return True when *name* belongs to a BatchNorm layer.

    Uses pre-computed BN prefixes derived from ``.running_mean``,
    ``.running_var``, and ``.num_batches_tracked`` parameters.
    """
    # Direct BN-specific parameter
    for suffix in _BN_SPECIFIC_SUFFIXES:
        if name.endswith(suffix):
            return True
    # .weight or .bias at a BN prefix
    for bn_prefix in bn_prefixes:
        if name.startswith(bn_prefix + "."):
            remaining = name[len(bn_prefix) + 1:]
            if remaining in ("weight", "bias"):
                return True
    return False


def fedbn_weighted_average_updates(updates: list[ClientUpdate]) -> Weights:
    """Aggregate all parameters *except* batch-norm statistics.

    Each client keeps its own BN running mean/var and affine parameters
    local — only the non-BN layers are averaged.  This is the core idea
    behind FedBN (Li et al., 2021): local batch-norm for feature-shift
    robustness.

    BN parameters are identified by their unique suffixes
    (``.running_mean``, ``.running_var``, ``.num_batches_tracked``) and
    the ``.weight`` / ``.bias`` sharing the same prefix.
    """
    _validate_updates(updates)
    reference = updates[0].weights
    total_examples = sum(update.num_examples for update in updates)
    bn_prefixes = _identify_batch_norm_parameter_prefixes(set(reference.keys()))
    result: Weights = {}

    for key, reference_value in reference.items():
        if _is_batch_norm_parameter(key, bn_prefixes):
            # Keep the first client's BN stats (deterministic but arbitrary).
            # In production each client would keep its own.
            result[key] = reference_value.copy()
            continue
        accumulator = np.zeros(reference_value.shape, dtype=np.float64)
        for update in updates:
            accumulator += update.weights[key].astype(np.float64) * update.num_examples
        averaged = accumulator / total_examples
        if np.issubdtype(reference_value.dtype, np.integer):
            averaged = np.rint(averaged)
        result[key] = averaged.astype(reference_value.dtype)
    return result


# ---------------------------------------------------------------------------
# SCAFFOLD – server control variate
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldServerState:
    """Mutable server-side SCAFFOLD control variate.

    ``c`` tracks the server control variate (same shape as global weights).
    ``c_i`` is the per-client control variate sent by each client.
    """

    c: Weights | None = None
    client_control_variates: dict[str, Weights] = field(default_factory=dict)

    def init_c(self, global_weights: Weights) -> None:
        """Initialise the server control variate to zeros matching *global_weights*."""
        self.c = {}
        for key, value in global_weights.items():
            self.c[key] = np.zeros(value.shape, dtype=np.float64)

    def update_c(
        self,
        updates: list[ClientUpdate],
        total_examples: int,
        server_lr: float = 1.0,
    ) -> None:
        """Update the global control variate after a round.

        c ← c + (server_lr / K) · Σᵢ (c_i - c)

        where c_i is the per-client delta reported by each client.
        """
        if self.c is None:
            raise RuntimeError("SCAFFOLD control variate not initialised")
        K = len(updates)
        delta: Weights = {}
        for key in self.c:
            delta_sum = np.zeros(self.c[key].shape, dtype=np.float64)
            for update in updates:
                c_i = update.weights.get(f"_scaffold_c_i_{key}")
                if c_i is not None:
                    delta_sum += c_i.astype(np.float64) - self.c[key]
            delta[key] = (server_lr / K) * delta_sum
        for key in self.c:
            self.c[key] = self.c[key] + delta[key]


def scaffold_weighted_average_updates(
    updates: list[ClientUpdate],
    server_state: ScaffoldServerState,
    server_lr: float = 1.0,
) -> Weights:
    """SCAFFOLD aggregation: weighted average + control variate correction.

    w_{t+1} = Σ (n_k / N) · w_{t+1}^k  +  server_lr · (c - c_avg)

    The correction term is folded into the returned weights.
    """
    _validate_updates(updates)
    reference = updates[0].weights
    total_examples = sum(update.num_examples for update in updates)
    result: Weights = {}

    if server_state.c is None:
        server_state.init_c(reference)

    # Compute average client control variate
    c_avg: Weights = {}
    for key in server_state.c:
        c_avg[key] = np.zeros(server_state.c[key].shape, dtype=np.float64)
        count = 0
        for update in updates:
            c_i = update.weights.get(f"_scaffold_c_i_{key}")
            if c_i is not None:
                c_avg[key] += c_i.astype(np.float64)
                count += 1
        if count > 0:
            c_avg[key] /= count

    for key, reference_value in reference.items():
        if key.startswith("_scaffold_"):
            result[key] = reference_value.copy()
            continue
        # Weighted average of client weights
        accumulator = np.zeros(reference_value.shape, dtype=np.float64)
        for update in updates:
            accumulator += update.weights[key].astype(np.float64) * update.num_examples
        averaged = accumulator / total_examples
        # SCAFFOLD correction: + ηₛ · (c - c_avg)
        if key in server_state.c and key in c_avg:
            correction = server_lr * (server_state.c[key] - c_avg[key])
            averaged += correction
        if np.issubdtype(reference_value.dtype, np.integer):
            averaged = np.rint(averaged)
        result[key] = averaged.astype(reference_value.dtype)

    # Update server control variate for next round
    server_state.update_c(updates, total_examples, server_lr)
    return result


# ---------------------------------------------------------------------------
# MOON – model-contrastive (aggregation is standard FedAvg)
# ---------------------------------------------------------------------------
# MOON uses a contrastive loss on the *client* side (cosine similarity
# between current representation, previous local model, and global model).
# The *server* aggregation is identical to FedAvg.  The client-side logic
# is implemented in the trainer module.
# ---------------------------------------------------------------------------


def moon_weighted_average_updates(updates: list[ClientUpdate]) -> Weights:
    """MOON aggregation: identical to FedAvg on the server side.

    The MOON contrastive loss is applied during local training, not at
    aggregation time.  This function is an alias for clarity.
    """
    return weighted_average_updates(updates)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def aggregate(
    updates: list[ClientUpdate],
    algorithm: str = "fedavg",
    *,
    scaffold_state: ScaffoldServerState | None = None,
    server_lr: float = 1.0,
) -> Weights:
    """Dispatch to the correct aggregation function."""
    algorithm = algorithm.lower()
    if algorithm == "fedavg":
        return weighted_average_updates(updates)
    if algorithm == "fedprox":
        return weighted_average_updates(updates)  # Same server aggregation
    if algorithm == "fedbn":
        return fedbn_weighted_average_updates(updates)
    if algorithm == "scaffold":
        state = scaffold_state or ScaffoldServerState()
        return scaffold_weighted_average_updates(updates, state, server_lr)
    if algorithm == "moon":
        return moon_weighted_average_updates(updates)
    raise ValueError(f"unknown algorithm: {algorithm!r}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
