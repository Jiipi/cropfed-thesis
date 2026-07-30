"""Normalize experiment result histories into query-friendly database rows."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete
from sqlmodel import Session

from cropfed.api.models import ClientRoundMetricRecord, ExperimentRoundRecord


def replace_round_history(
    session: Session,
    experiment_id: str,
    result: dict[str, Any],
) -> int:
    """Replace normalized round rows using the canonical history in a result."""

    history = result.get("history", [])
    if not isinstance(history, list):
        raise ValueError("experiment result history must be a list")

    session.exec(
        delete(ExperimentRoundRecord).where(
            ExperimentRoundRecord.experiment_id == experiment_id
        )
    )
    seen_rounds: set[int] = set()
    for payload in history:
        if not isinstance(payload, dict):
            raise ValueError("every experiment history entry must be an object")
        round_number = _round_number(payload)
        if round_number in seen_rounds:
            raise ValueError(f"duplicate experiment round {round_number}")
        seen_rounds.add(round_number)
        summary = summarize_round(payload)
        session.add(
            ExperimentRoundRecord(
                experiment_id=experiment_id,
                round_number=round_number,
                metrics_json=json.dumps(payload, ensure_ascii=False),
                **summary,
            )
        )
    return len(history)


def replace_client_history(
    session: Session,
    experiment_id: str,
    result: dict[str, Any],
) -> int:
    """Replace per-client Flower rows; synthetic results legitimately have none."""

    history = result.get("client_history", [])
    if not isinstance(history, list):
        raise ValueError("experiment client_history must be a list")
    session.exec(
        delete(ClientRoundMetricRecord).where(
            ClientRoundMetricRecord.experiment_id == experiment_id
        )
    )
    seen: set[tuple[int, int, str]] = set()
    for payload in history:
        if not isinstance(payload, dict):
            raise ValueError("every client history entry must be an object")
        round_number = _positive_integer(payload, "round")
        client_id = _non_negative_integer(payload, "client_id")
        phase = payload.get("phase")
        if phase not in {"train", "evaluate"}:
            raise ValueError("client history phase must be 'train' or 'evaluate'")
        identity = (round_number, client_id, phase)
        if identity in seen:
            raise ValueError(f"duplicate client history entry {identity}")
        seen.add(identity)
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("client history metrics must be an object")
        session.add(
            ClientRoundMetricRecord(
                experiment_id=experiment_id,
                round_number=round_number,
                client_id=client_id,
                phase=phase,
                node_id=str(payload.get("node_id", "")),
                num_examples=_non_negative_integer(payload, "num_examples"),
                metrics_json=json.dumps(metrics, ensure_ascii=False),
                payload_download_bytes=_non_negative_integer(
                    payload, "payload_download_bytes"
                ),
                payload_upload_bytes=_non_negative_integer(
                    payload, "payload_upload_bytes"
                ),
                model_download_bytes=_non_negative_integer(
                    payload, "model_download_bytes"
                ),
                model_upload_bytes=_non_negative_integer(
                    payload, "model_upload_bytes"
                ),
            )
        )
    return len(history)


def summarize_round(payload: dict[str, Any]) -> dict[str, float | int | None]:
    """Extract common columns from synthetic and Flower history shapes."""

    central = _mapping(payload.get("central_evaluate"))
    federated = _mapping(payload.get("federated_evaluate"))
    train = _mapping(payload.get("train"))
    communication = _mapping(payload.get("communication"))
    evaluation = central or federated or payload

    return {
        "train_loss": _first_number(train, "train_loss")
        if train
        else _first_number(payload, "train_loss"),
        "evaluation_loss": _first_number(
            evaluation,
            "central_loss",
            "eval_loss",
            "loss",
            "test_loss",
        ),
        "accuracy": _first_number(
            evaluation,
            "central_accuracy",
            "eval_accuracy",
            "accuracy",
        ),
        "macro_f1": _first_number(
            evaluation,
            "central_macro_f1",
            "eval_macro_f1",
            "macro_f1",
        ),
        "harmful_missed_as_healthy_rate": _first_number(
            evaluation,
            "central_harmful_missed_as_healthy_rate",
            "eval_harmful_missed_as_healthy_rate",
            "harmful_missed_as_healthy_rate",
        ),
        "elapsed_seconds": _first_number(payload, "round_seconds", "elapsed_seconds"),
        "bytes_up": _first_integer(
            communication, "payload_upload_bytes"
        )
        or _first_integer(payload, "bytes_up"),
        "bytes_down": _first_integer(
            communication, "payload_download_bytes"
        )
        or _first_integer(payload, "bytes_down"),
    }


def round_payload(record: ExperimentRoundRecord) -> dict[str, Any]:
    """Deserialize the complete payload retained for an experiment round."""

    payload = json.loads(record.metrics_json)
    if not isinstance(payload, dict):
        raise ValueError("stored round metrics must decode to an object")
    return payload


def round_summary(record: ExperimentRoundRecord) -> dict[str, float | int | None]:
    """Return stable scalar columns for charting and result exports."""

    return {
        "round": record.round_number,
        "train_loss": record.train_loss,
        "evaluation_loss": record.evaluation_loss,
        "accuracy": record.accuracy,
        "macro_f1": record.macro_f1,
        "harmful_missed_as_healthy_rate": record.harmful_missed_as_healthy_rate,
        "elapsed_seconds": record.elapsed_seconds,
        "bytes_up": record.bytes_up,
        "bytes_down": record.bytes_down,
    }


def client_metric_payload(record: ClientRoundMetricRecord) -> dict[str, Any]:
    """Return a stable API shape for one client/round/phase row."""

    metrics = json.loads(record.metrics_json)
    if not isinstance(metrics, dict):
        raise ValueError("stored client metrics must decode to an object")
    return {
        "round": record.round_number,
        "client_id": record.client_id,
        "phase": record.phase,
        "node_id": record.node_id,
        "num_examples": record.num_examples,
        "metrics": metrics,
        "payload_download_bytes": record.payload_download_bytes,
        "payload_upload_bytes": record.payload_upload_bytes,
        "model_download_bytes": record.model_download_bytes,
        "model_upload_bytes": record.model_upload_bytes,
    }


def _round_number(payload: dict[str, Any]) -> int:
    value = payload.get("round")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("experiment history round must be a non-negative integer")
    return value


def _positive_integer(payload: dict[str, Any], key: str) -> int:
    value = _non_negative_integer(payload, key)
    if value < 1:
        raise ValueError(f"client history {key} must be positive")
    return value


def _non_negative_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"client history {key} must be a non-negative integer")
    return value


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return None


def _first_integer(mapping: dict[str, Any], *keys: str) -> int | None:
    value = _first_number(mapping, *keys)
    return int(value) if value is not None else None
