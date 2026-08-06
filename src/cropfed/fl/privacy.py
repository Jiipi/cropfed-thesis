"""Differential Privacy utilities for Federated Learning.

Implements DP-SGD style gradient clipping and noise addition that can be
applied to client model updates before they are sent to the server.

The implementation follows the DP-FedAvg / DP-FL approach:
- Clip per-sample or per-client gradients to a maximum L2 norm
- Add calibrated Gaussian noise
- Track the privacy budget (ε, δ) via the moments accountant

References
----------
- Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
- McMahan et al., "Learning Differentially Private Recurrent Language Models", ICLR 2018
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any


@dataclass(frozen=True, slots=True)
class PrivacyBudget:
    """Track (ε, δ)-differential privacy consumption."""

    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    num_steps: int
    sample_rate: float

    @property
    def is_valid(self) -> bool:
        """Return True when the privacy budget is within reasonable bounds."""
        return self.epsilon > 0 and 0 < self.delta < 1


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    num_steps: int,
    sample_rate: float,
    *,
    max_epsilon: float = 50.0,
) -> float:
    """Estimate the Gaussian noise multiplier needed for (ε, δ)-DP.

    Uses the moments accountant approximation (Abadi et al. 2016).
    This is a simplified closed-form estimate; for production use a
    full RDP accountant.

    Parameters
    ----------
    target_epsilon:
        Target privacy budget ε (e.g. 8.0).
    target_delta:
        Target δ (e.g. 1e-5). Must be less than 1/N.
    num_steps:
        Total number of training steps (rounds × epochs × batches).
    sample_rate:
        Sampling probability per step (batch_size / dataset_size).
    max_epsilon:
        Upper bound on ε to prevent infinite search.

    Returns
    -------
    float
        The noise multiplier σ such that adding N(0, σ²·C²) to clipped
        gradients achieves roughly (ε, δ)-DP.
    """
    if target_epsilon <= 0 or target_epsilon > max_epsilon:
        raise ValueError(f"target_epsilon must be in (0, {max_epsilon}]")
    if target_delta <= 0 or target_delta >= 1:
        raise ValueError("target_delta must be in (0, 1)")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")

    # Binary search for noise multiplier
    low, high = 0.01, 100.0
    for _ in range(50):
        mid = (low + high) / 2
        eps = _estimate_epsilon(mid, sample_rate, num_steps, target_delta)
        if eps > target_epsilon:
            low = mid
        else:
            high = mid
    return high


def _estimate_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    delta: float,
) -> float:
    """Simplified moments accountant estimate."""
    # q = sample_rate
    q = sample_rate
    # Privacy amplification by sampling: effective ε per step
    if noise_multiplier < 1e-8:
        return float("inf")

    # Use the tight composition theorem approximation
    # For Gaussian mechanism with sampling:
    # α(λ) ≈ q² · λ · (λ + 1) / (2 · σ²)  for small q
    # We compute the optimal λ that minimizes ε

    best_epsilon = float("inf")
    for lam in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
        # Moment at order λ
        if q <= 0 or noise_multiplier <= 0:
            continue
        moment = (q * q * lam * (lam + 1)) / (2 * noise_multiplier * noise_multiplier)
        # Total moment after num_steps
        total_moment = num_steps * moment
        # Convert to (ε, δ) via the tail bound
        epsilon = total_moment + (log(1.0 / delta) / (lam - 1)) if lam > 1 else float("inf")
        if epsilon < best_epsilon:
            best_epsilon = epsilon

    return best_epsilon


def clip_gradients(
    parameters: dict[str, Any],
    max_norm: float,
) -> tuple[dict[str, Any], float]:
    """Clip parameter gradients or deltas to a maximum L2 norm.

    Parameters
    ----------
    parameters:
        Dictionary of parameter name to tensor (numpy or torch).
    max_norm:
        Maximum L2 norm for clipping.

    Returns
    -------
    tuple[dict, float]
        (clipped parameters, total_norm_before_clipping)
    """
    import numpy as np

    total_norm_sq = 0.0
    for value in parameters.values():
        arr = np.asarray(value, dtype=np.float64)
        total_norm_sq += float(np.sum(arr * arr))

    total_norm = sqrt(total_norm_sq)
    if total_norm <= max_norm:
        return parameters, total_norm

    scale = max_norm / total_norm
    clipped = {
        name: (np.asarray(value, dtype=np.float64) * scale).astype(
            np.asarray(value).dtype
        )
        for name, value in parameters.items()
    }
    return clipped, total_norm


def add_gaussian_noise(
    parameters: dict[str, Any],
    noise_multiplier: float,
    max_norm: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Add calibrated Gaussian noise to parameters.

    The noise scale is noise_multiplier * max_norm, which is the standard
    deviation of the Gaussian noise added to each parameter.

    Parameters
    ----------
    parameters:
        Dictionary of parameter name to tensor.
    noise_multiplier:
        σ in the Gaussian mechanism.
    max_norm:
        Clipping norm C. Noise std = σ * C.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    dict
        Parameters with added Gaussian noise.
    """
    import numpy as np

    noise_std = noise_multiplier * max_norm
    rng = np.random.default_rng(seed)
    noisy = {}
    for name, value in parameters.items():
        arr = np.asarray(value, dtype=np.float64)
        noise = rng.normal(0.0, noise_std, size=arr.shape)
        noisy[name] = (arr + noise).astype(np.asarray(value).dtype)
    return noisy


def apply_dp_to_client_update(
    weights: dict[str, Any],
    max_grad_norm: float,
    noise_multiplier: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Apply DP clipping + noise to a single client model update.

    This is the standard DP-FL pattern: clip the delta (w_local - w_global),
    add noise, and return the protected weights.

    Parameters
    ----------
    weights:
        Client model weights (typically the delta from global).
    max_grad_norm:
        Clipping threshold C.
    noise_multiplier:
        Gaussian noise multiplier σ.
    seed:
        Optional RNG seed.

    Returns
    -------
    dict
        DP-protected weights.
    """
    clipped, _ = clip_gradients(weights, max_grad_norm)
    return add_gaussian_noise(clipped, noise_multiplier, max_grad_norm, seed)


def privacy_accountant_summary(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    delta: float = 1e-5,
) -> dict[str, Any]:
    """Return a human-readable privacy budget summary."""
    epsilon = _estimate_epsilon(noise_multiplier, sample_rate, num_steps, delta)
    return {
        "mechanism": "Gaussian",
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": "C (applied per-client)",
        "sample_rate": sample_rate,
        "num_steps": num_steps,
        "target_delta": delta,
        "estimated_epsilon": epsilon,
        "is_meaningful": epsilon < 50.0,
        "note": (
            "Simplified moments accountant estimate. For publication, "
            "use a full RDP accountant (e.g. opacus, tf-privacy)."
        ),
    }