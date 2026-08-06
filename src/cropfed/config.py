"""Validated experiment configuration independent of any web framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Algorithm = Literal["fedavg", "fedprox", "fedbn", "scaffold", "moon"]
PartitionKind = Literal["iid", "dirichlet", "quantity_skew", "feature_skew"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration shared by the smoke simulator and real FL runner."""

    algorithm: Algorithm = "fedavg"
    partition_kind: PartitionKind = "dirichlet"
    num_clients: int = 4
    num_rounds: int = 5
    local_epochs: int = 2
    learning_rate: float = 0.05
    batch_size: int = 32
    dirichlet_alpha: float = 0.5
    proximal_mu: float = 0.01
    # SCAFFOLD server learning rate (ηₛ / η_global)
    scaffold_server_lr: float = 1.0
    # MOON contrastive loss temperature τ
    moon_temperature: float = 0.5
    # MOON contrastive loss weight μ_moon
    moon_mu: float = 1.0
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.algorithm not in {"fedavg", "fedprox", "fedbn", "scaffold", "moon"}:
            raise ValueError(
                "algorithm must be 'fedavg', 'fedprox', 'fedbn', 'scaffold', or 'moon'"
            )
        if self.partition_kind not in {
            "iid",
            "dirichlet",
            "quantity_skew",
            "feature_skew",
        }:
            raise ValueError(
                "partition_kind must be 'iid', 'dirichlet', 'quantity_skew', "
                "or 'feature_skew'"
            )
        if self.num_clients < 2:
            raise ValueError("num_clients must be at least 2")
        if self.num_rounds < 1 or self.local_epochs < 1:
            raise ValueError("num_rounds and local_epochs must be positive")
        if self.learning_rate <= 0 or self.batch_size < 1:
            raise ValueError("learning_rate and batch_size must be positive")
        if self.partition_kind == "dirichlet" and self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if self.proximal_mu < 0:
            raise ValueError("proximal_mu cannot be negative")
        if self.scaffold_server_lr <= 0:
            raise ValueError("scaffold_server_lr must be positive")
        if self.moon_temperature <= 0:
            raise ValueError("moon_temperature must be positive")
        if self.moon_mu < 0:
            raise ValueError("moon_mu cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

