"""Convert rich image-classification evaluation into Flower-safe scalar/list values."""

from __future__ import annotations

from typing import Any

from cropfed.constants import TOMATO_CLASSES


def flower_evaluation_values(
    evaluation,
    *,
    prefix: str,
    detailed: bool,
) -> dict[str, int | float | list[int] | list[float]]:
    """Flatten evaluation output without losing the harmful-as-healthy signal."""

    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("metric prefix must be a non-empty identifier")
    metrics: dict[str, Any] = evaluation.metrics
    per_class = metrics["per_class"]
    group_metrics = metrics["group_metrics"]["per_class"]
    values: dict[str, int | float | list[int] | list[float]] = {
        f"{prefix}_loss": float(evaluation.loss),
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_macro_precision": float(metrics["macro_precision"]),
        f"{prefix}_macro_recall": float(metrics["macro_recall"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_harmful_missed_as_healthy_rate": float(
            metrics["harmful_missed_as_healthy_rate"]
        ),
        f"{prefix}_harmful_detection_recall": float(
            metrics["harmful_detection_recall"]
        ),
        f"{prefix}_disease_f1": float(group_metrics["disease"]["f1"]),
        f"{prefix}_pest_f1": float(group_metrics["pest"]["f1"]),
        f"{prefix}_spider_mite_f1": float(
            per_class["Two-spotted spider mite"]["f1"]
        ),
    }
    if detailed:
        values.update(
            {
                f"{prefix}_harmful_missed_as_healthy_count": int(
                    metrics["harmful_missed_as_healthy_count"]
                ),
                f"{prefix}_per_class_recall": [
                    float(per_class[name]["recall"]) for name in TOMATO_CLASSES
                ],
                f"{prefix}_per_class_precision": [
                    float(per_class[name]["precision"]) for name in TOMATO_CLASSES
                ],
                f"{prefix}_per_class_f1": [
                    float(per_class[name]["f1"]) for name in TOMATO_CLASSES
                ],
                f"{prefix}_per_class_support": [
                    int(per_class[name]["support"]) for name in TOMATO_CLASSES
                ],
                f"{prefix}_confusion_matrix_flat": [
                    int(value)
                    for row in metrics["confusion_matrix"]
                    for value in row
                ],
                f"{prefix}_confusion_matrix_size": len(TOMATO_CLASSES),
            }
        )
    return values
