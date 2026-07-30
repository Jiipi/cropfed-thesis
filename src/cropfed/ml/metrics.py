"""Dependency-light multiclass classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> NDArray[np.int64]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1 or true.size != pred.size:
        raise ValueError("y_true and y_pred must be one-dimensional and equal length")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if true.size and (
        np.any(true < 0)
        or np.any(pred < 0)
        or np.any(true >= num_classes)
        or np.any(pred >= num_classes)
    ):
        raise ValueError("labels are outside [0, num_classes)")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    class_names: Sequence[str] | None = None,
    healthy_class_id: int | None = None,
    class_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy and macro/per-class precision, recall and F1."""

    matrix = confusion_matrix(y_true, y_pred, num_classes)
    names = list(class_names or [str(index) for index in range(num_classes)])
    if len(names) != num_classes:
        raise ValueError("class_names length must equal num_classes")

    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual != 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    total = int(matrix.sum())

    result: dict[str, Any] = {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(actual[index]),
            }
            for index in range(num_classes)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    if healthy_class_id is not None:
        if healthy_class_id < 0 or healthy_class_id >= num_classes:
            raise ValueError("healthy_class_id is outside the class range")
        harmful_mask = np.ones(num_classes, dtype=bool)
        harmful_mask[healthy_class_id] = False
        harmful_total = int(actual[harmful_mask].sum())
        missed_as_healthy = int(matrix[harmful_mask, healthy_class_id].sum())
        missed_rate = missed_as_healthy / harmful_total if harmful_total else 0.0
        result["harmful_missed_as_healthy_count"] = missed_as_healthy
        result["harmful_missed_as_healthy_rate"] = float(missed_rate)
        result["harmful_detection_recall"] = float(1.0 - missed_rate)

    if class_groups is not None:
        if len(class_groups) != num_classes:
            raise ValueError("class_groups length must equal num_classes")
        group_names = list(dict.fromkeys(class_groups))
        group_index = {name: index for index, name in enumerate(group_names)}
        mapping = np.asarray([group_index[name] for name in class_groups], dtype=np.int64)
        true = np.asarray(y_true, dtype=np.int64)
        pred = np.asarray(y_pred, dtype=np.int64)
        result["group_metrics"] = classification_metrics(
            mapping[true],
            mapping[pred],
            num_classes=len(group_names),
            class_names=group_names,
        )
    return result
