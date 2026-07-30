"""Small NumPy federation used only to validate infrastructure and FL logic.

This module does not produce thesis results. It intentionally uses synthetic
feature vectors rather than images so that CI and a new developer can verify
partitioning, local training, FedAvg/FedProx and metric logging without a GPU or
the PlantVillage dataset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cropfed.config import ExperimentConfig
from cropfed.data.partitioning import make_partitions, partition_statistics
from cropfed.fl.aggregation import ClientUpdate, weighted_average_updates
from cropfed.ml.metrics import classification_metrics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    train_x: FloatArray
    train_y: NDArray[np.int64]
    test_x: FloatArray
    test_y: NDArray[np.int64]


def make_synthetic_dataset(
    num_classes: int = 10,
    samples_per_class: int = 60,
    num_features: int = 16,
    seed: int = 2026,
) -> SyntheticDataset:
    """Generate a deterministic, linearly separable multiclass dataset."""

    if num_classes < 2 or samples_per_class < 10 or num_features < 2:
        raise ValueError("synthetic dataset dimensions are too small")
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 2.5, size=(num_classes, num_features))
    train_x: list[FloatArray] = []
    train_y: list[NDArray[np.int64]] = []
    test_x: list[FloatArray] = []
    test_y: list[NDArray[np.int64]] = []

    test_count = max(2, samples_per_class // 5)
    for class_id in range(num_classes):
        samples = centers[class_id] + rng.normal(
            0.0, 0.75, size=(samples_per_class, num_features)
        )
        order = rng.permutation(samples_per_class)
        test_indices = order[:test_count]
        train_indices = order[test_count:]
        test_x.append(samples[test_indices])
        train_x.append(samples[train_indices])
        test_y.append(np.full(test_count, class_id, dtype=np.int64))
        train_y.append(np.full(samples_per_class - test_count, class_id, dtype=np.int64))

    return SyntheticDataset(
        train_x=np.vstack(train_x).astype(np.float64),
        train_y=np.concatenate(train_y),
        test_x=np.vstack(test_x).astype(np.float64),
        test_y=np.concatenate(test_y),
    )


def run_synthetic_experiment(
    config: ExperimentConfig,
    *,
    samples_per_class: int = 60,
    num_features: int = 16,
) -> dict[str, Any]:
    """Run a complete synthetic federation and return JSON-serializable results."""

    started = time.perf_counter()
    dataset = make_synthetic_dataset(
        samples_per_class=samples_per_class,
        num_features=num_features,
        seed=config.seed,
    )
    num_classes = int(dataset.train_y.max()) + 1
    train_partitions = make_partitions(
        dataset.train_y,
        config.num_clients,
        config.partition_kind,
        config.dirichlet_alpha,
        config.seed,
    )
    test_partitions = make_partitions(
        dataset.test_y,
        config.num_clients,
        config.partition_kind,
        config.dirichlet_alpha,
        config.seed + 1,
    )

    rng = np.random.default_rng(config.seed)
    global_weights: dict[str, NDArray[np.generic]] = {
        "weight": rng.normal(0.0, 0.01, size=(num_features, num_classes)),
        "bias": np.zeros(num_classes, dtype=np.float64),
    }
    model_bytes = int(sum(array.nbytes for array in global_weights.values()))
    history: list[dict[str, Any]] = []

    for round_number in range(1, config.num_rounds + 1):
        round_started = time.perf_counter()
        updates: list[ClientUpdate] = []
        for client_id, indices in enumerate(train_partitions):
            local_weights, train_loss = _train_local_softmax(
                dataset.train_x[indices],
                dataset.train_y[indices],
                global_weights,
                epochs=config.local_epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                proximal_mu=config.proximal_mu if config.algorithm == "fedprox" else 0.0,
                seed=config.seed + (round_number * 1_000) + client_id,
            )
            updates.append(
                ClientUpdate(
                    client_id=str(client_id),
                    weights=local_weights,
                    num_examples=int(indices.size),
                    metrics={"train_loss": train_loss},
                )
            )
        global_weights = weighted_average_updates(updates)
        predictions = _predict(dataset.test_x, global_weights)
        global_metrics = classification_metrics(dataset.test_y, predictions, num_classes)

        client_f1: list[float] = []
        for indices in test_partitions:
            client_predictions = _predict(dataset.test_x[indices], global_weights)
            metrics = classification_metrics(
                dataset.test_y[indices], client_predictions, num_classes
            )
            client_f1.append(float(metrics["macro_f1"]))

        history.append(
            {
                "round": round_number,
                "train_loss": float(
                    sum(
                        update.metrics["train_loss"] * update.num_examples
                        for update in updates
                    )
                    / sum(update.num_examples for update in updates)
                ),
                "accuracy": global_metrics["accuracy"],
                "macro_precision": global_metrics["macro_precision"],
                "macro_recall": global_metrics["macro_recall"],
                "macro_f1": global_metrics["macro_f1"],
                "worst_client_macro_f1": min(client_f1),
                "client_macro_f1": client_f1,
                "round_seconds": time.perf_counter() - round_started,
                "bytes_up": model_bytes * config.num_clients,
                "bytes_down": model_bytes * config.num_clients,
            }
        )

    final_predictions = _predict(dataset.test_x, global_weights)
    return {
        "result_kind": "synthetic_smoke_only",
        "warning": (
            "Synthetic vectors validate code paths only; these values must not be "
            "reported as crop-disease experiment results."
        ),
        "config": config.to_dict(),
        "data": {
            "num_train": int(dataset.train_y.size),
            "num_test": int(dataset.test_y.size),
            "num_classes": num_classes,
            "num_features": num_features,
            "train_partitions": partition_statistics(
                dataset.train_y, train_partitions, num_classes
            ),
        },
        "communication": {
            "model_bytes": model_bytes,
            "total_bytes": int(
                sum(row["bytes_up"] + row["bytes_down"] for row in history)
            ),
        },
        "history": history,
        "final_metrics": classification_metrics(
            dataset.test_y, final_predictions, num_classes
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _train_local_softmax(
    features: FloatArray,
    labels: NDArray[np.int64],
    global_weights: dict[str, NDArray[np.generic]],
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    proximal_mu: float,
    seed: int,
) -> tuple[dict[str, NDArray[np.generic]], float]:
    weight = global_weights["weight"].astype(np.float64, copy=True)
    bias = global_weights["bias"].astype(np.float64, copy=True)
    reference_weight = weight.copy()
    reference_bias = bias.copy()
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for _ in range(epochs):
        for start in range(0, labels.size, batch_size):
            if start == 0:
                order = rng.permutation(labels.size)
            indices = order[start : start + batch_size]
            batch_x = features[indices]
            batch_y = labels[indices]
            logits = batch_x @ weight + bias
            probabilities = _softmax(logits)
            clipped = np.clip(probabilities[np.arange(batch_y.size), batch_y], 1e-12, 1.0)
            cross_entropy = -float(np.log(clipped).mean())
            proximal = 0.5 * proximal_mu * (
                float(np.square(weight - reference_weight).sum())
                + float(np.square(bias - reference_bias).sum())
            )
            losses.append(cross_entropy + proximal)

            gradient_logits = probabilities
            gradient_logits[np.arange(batch_y.size), batch_y] -= 1.0
            gradient_logits /= batch_y.size
            gradient_weight = batch_x.T @ gradient_logits
            gradient_bias = gradient_logits.sum(axis=0)
            if proximal_mu:
                gradient_weight += proximal_mu * (weight - reference_weight)
                gradient_bias += proximal_mu * (bias - reference_bias)
            weight -= learning_rate * gradient_weight
            bias -= learning_rate * gradient_bias

    return {"weight": weight, "bias": bias}, float(np.mean(losses))


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _predict(
    features: FloatArray, weights: dict[str, NDArray[np.generic]]
) -> NDArray[np.int64]:
    logits = features @ weights["weight"] + weights["bias"]
    return np.argmax(logits, axis=1).astype(np.int64)
