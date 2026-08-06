"""Convert rich image-classification evaluation into Flower-safe scalar/list values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _group_f1(group_metrics: Mapping[str, Any], group: str, prefix: str) -> float:
    """Read one group's F1, naming the taxonomy mismatch if the group is absent.

    ``group_metrics`` only contains the groups present in ``class_groups``, so a
    missing key means the caller's taxonomy disagrees with the evaluated labels
    rather than that the score is genuinely zero.
    """

    try:
        return float(group_metrics[group]["f1"])
    except KeyError:
        available = ", ".join(sorted(group_metrics)) or "none"
        raise KeyError(
            f"evaluation has no {group!r} class group for metric "
            f"{prefix}_{group}_f1; groups present: {available}"
        ) from None


def flower_evaluation_values(
    evaluation,
    *,
    prefix: str,
    detailed: bool,
    class_names: Sequence[str],
) -> dict[str, int | float | list[int] | list[float]]:
    """Flatten evaluation output without losing the harmful-as-healthy signal."""

    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("metric prefix must be a non-empty identifier")
    metrics: dict[str, Any] = evaluation.metrics
    resolved_class_names = tuple(class_names)
    per_class = metrics["per_class"]
    group_metrics = metrics["group_metrics"]["per_class"]
    spider_mite_name = next(
        (name for name in resolved_class_names if "spider mite" in name.lower()),
        None,
    )
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
        f"{prefix}_disease_f1": _group_f1(group_metrics, "disease", prefix),
        # PlantVillage labels exactly one pest class, so under the 38-class
        # taxonomy this group metric describes a single class out of 38 and is
        # not comparable to the ten-class tomato pilot figure.
        f"{prefix}_pest_f1": _group_f1(group_metrics, "pest", prefix),
        f"{prefix}_spider_mite_f1": float(
            per_class[spider_mite_name]["f1"] if spider_mite_name else 0.0
        ),
    }
    if detailed:
        values.update(
            {
                f"{prefix}_harmful_missed_as_healthy_count": int(
                    metrics["harmful_missed_as_healthy_count"]
                ),
                f"{prefix}_per_class_recall": [
                    float(per_class[name]["recall"]) for name in resolved_class_names
                ],
                f"{prefix}_per_class_precision": [
                    float(per_class[name]["precision"])
                    for name in resolved_class_names
                ],
                f"{prefix}_per_class_f1": [
                    float(per_class[name]["f1"]) for name in resolved_class_names
                ],
                f"{prefix}_per_class_support": [
                    int(per_class[name]["support"]) for name in resolved_class_names
                ],
                f"{prefix}_confusion_matrix_flat": [
                    int(value)
                    for row in metrics["confusion_matrix"]
                    for value in row
                ],
                f"{prefix}_confusion_matrix_size": len(resolved_class_names),
            }
        )
    return values
