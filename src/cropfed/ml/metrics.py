"""Dependency-light multiclass classification metrics."""

from __future__ import annotations

import math
import statistics
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
    healthy_class_ids: Sequence[int] | None = None,
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
    if healthy_class_id is not None and healthy_class_ids is not None:
        raise ValueError("provide healthy_class_id or healthy_class_ids, not both")
    resolved_healthy_ids = (
        tuple(healthy_class_ids)
        if healthy_class_ids is not None
        else (healthy_class_id,)
        if healthy_class_id is not None
        else ()
    )
    if resolved_healthy_ids:
        if len(set(resolved_healthy_ids)) != len(resolved_healthy_ids) or any(
            class_id < 0 or class_id >= num_classes
            for class_id in resolved_healthy_ids
        ):
            raise ValueError("healthy class IDs are invalid")
        harmful_mask = np.ones(num_classes, dtype=bool)
        harmful_mask[list(resolved_healthy_ids)] = False
        harmful_total = int(actual[harmful_mask].sum())
        missed_as_healthy = int(
            matrix[
                np.ix_(np.flatnonzero(harmful_mask), list(resolved_healthy_ids))
            ].sum()
        )
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


def client_fairness(
    scores: Sequence[float],
    *,
    num_examples: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarise how unevenly one metric is spread across clients.

    The proposal (§7) asks for "độ lệch accuracy giữa cơ sở mạnh và yếu" and
    warns that a good federation must not abandon small facilities. That makes
    two conventions non-negotiable here, and both are the opposite of what the
    aggregate metrics do:

    * The mean is **unweighted** — one client, one vote. Weighting by sample
      count is what hides an abandoned small client, because the clients being
      abandoned are precisely the ones with little data. ``weighted_mean`` is
      reported alongside it so the two can be compared: when it sits well above
      ``mean``, the large clients are carrying the score.
    * The deviation is the **population** standard deviation. These clients are
      the whole federation, not a sample drawn from one, so dividing by ``n-1``
      would inflate the number by 15% at four clients and make the run look
      less fair than it is.
    """

    values = [float(score) for score in scores]
    if not values:
        raise ValueError("fairness needs at least one client score")
    if any(math.isnan(value) or math.isinf(value) for value in values):
        raise ValueError("client scores must be finite")

    mean = statistics.fmean(values)
    worst = min(values)
    best = max(values)
    result: dict[str, Any] = {
        "num_clients": len(values),
        "mean": mean,
        "worst": worst,
        "best": best,
        # Population standard deviation: these clients are the federation.
        "std": statistics.pstdev(values),
        "spread": best - worst,
        # Scale-free, so an accuracy spread and an F1 spread are comparable —
        # but undefined at mean 0, where the ratio carries no information.
        "coefficient_of_variation": (
            statistics.pstdev(values) / mean if mean > 0 else None
        ),
    }

    if num_examples is None:
        return result

    sizes = [int(count) for count in num_examples]
    if len(sizes) != len(values):
        raise ValueError("num_examples must have one entry per client score")
    if any(count < 0 for count in sizes):
        raise ValueError("num_examples cannot be negative")
    total = sum(sizes)
    if total == 0:
        raise ValueError("num_examples must not sum to zero")

    weighted_mean = (
        sum(value * count for value, count in zip(values, sizes, strict=True)) / total
    )
    smallest = min(range(len(sizes)), key=lambda index: sizes[index])
    largest = max(range(len(sizes)), key=lambda index: sizes[index])
    result.update(
        {
            "weighted_mean": weighted_mean,
            # Positive means the big clients score above the federation average,
            # which is the shape of "small facilities left behind" (§7).
            "size_advantage": weighted_mean - mean,
            "smallest_client_score": values[smallest],
            "largest_client_score": values[largest],
            "smallest_client_examples": sizes[smallest],
            "largest_client_examples": sizes[largest],
        }
    )
    return result


def gap_vs_centralized(
    federated_score: float | None,
    centralized_score: float | None,
) -> float | None:
    """Return how far a federated score falls short of the centralized one.

    Signed as ``centralized - federated`` so that a **positive** gap always
    means the federation is behind, matching §3's "khoảng cách accuracy nhỏ,
    ví dụ dưới vài phần trăm". The sign is worth stating in one place: an
    inverted gap column would read as the federation beating centralized
    training, which is the thesis's central claim pointing backwards.

    ``None`` when either side is missing — never ``0.0``, which would claim the
    federation exactly matched a baseline that was never run.
    """

    if federated_score is None or centralized_score is None:
        return None
    return float(centralized_score) - float(federated_score)
