"""Flower strategies with auditable per-client metrics and payload byte counts."""

from __future__ import annotations

import time
from collections.abc import Iterable
from math import isfinite
from typing import Any

import numpy as np
from flwr.app import Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg, FedProx

MAX_CLIENT_METADATA_BYTES = 1_000_000


def recorddict_payload_bytes(content: RecordDict) -> int:
    """Count serialized record payload bytes, excluding transport framing/TLS."""

    records = (
        list(content.array_records.values())
        + list(content.metric_records.values())
        + list(content.config_records.values())
    )
    return int(sum(record.count_bytes() for record in records))


def recorddict_array_bytes(content: RecordDict) -> int:
    """Count only model/array payload bytes in a Flower RecordDict."""

    return int(sum(record.count_bytes() for record in content.array_records.values()))


class _TrackingMixin:
    """Capture messages without changing FedAvg/FedProx aggregation semantics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.client_history: list[dict[str, Any]] = []
        self._sent_messages: dict[tuple[int, str, int], dict[str, Any]] = {}
        self._phase_started: dict[tuple[int, str], float] = {}

    def configure_train(self, server_round, arrays, config, grid):
        self._phase_started[(server_round, "train")] = time.perf_counter()
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self._track_sent(server_round, "train", messages)
        return messages

    def configure_evaluate(self, server_round, arrays, config, grid):
        self._phase_started[(server_round, "evaluate")] = time.perf_counter()
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        self._track_sent(server_round, "evaluate", messages)
        return messages

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        reply_list = list(replies)
        communication = self._track_replies(server_round, "train", reply_list)
        arrays, metrics = super().aggregate_train(server_round, reply_list)
        if metrics is not None:
            metrics.pop("client-id", None)
            _add_communication_metrics(metrics, communication)
            metrics["phase_seconds"] = self._phase_elapsed(server_round, "train")
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        reply_list = list(replies)
        communication = self._track_replies(server_round, "evaluate", reply_list)
        metrics = super().aggregate_evaluate(server_round, reply_list)
        if metrics is not None:
            metrics.pop("client-id", None)
            _add_communication_metrics(metrics, communication)
            metrics["phase_seconds"] = self._phase_elapsed(server_round, "evaluate")
        return metrics

    def _track_sent(
        self,
        server_round: int,
        phase: str,
        messages: Iterable[Message],
    ) -> None:
        for message in messages:
            node_id = int(message.metadata.dst_node_id)
            self._sent_messages[(server_round, phase, node_id)] = {
                "payload_download_bytes": recorddict_payload_bytes(message.content),
                "model_download_bytes": recorddict_array_bytes(message.content),
                "array_schema": _array_schema(message.content),
            }

    def _track_replies(
        self,
        server_round: int,
        phase: str,
        replies: Iterable[Message],
    ) -> dict[str, int]:
        totals = {
            "payload_download_bytes": 0,
            "payload_upload_bytes": 0,
            "model_download_bytes": 0,
            "model_upload_bytes": 0,
        }
        sent_for_round = {
            node_id: values
            for (round_number, sent_phase, node_id), values in self._sent_messages.items()
            if round_number == server_round and sent_phase == phase
        }
        for sent in sent_for_round.values():
            totals["payload_download_bytes"] += sent["payload_download_bytes"]
            totals["model_download_bytes"] += sent["model_download_bytes"]

        seen_client_ids: set[int] = set()
        seen_node_ids: set[int] = set()
        for message in replies:
            if message.has_error() or not message.has_content():
                continue
            node_id = int(message.metadata.src_node_id)
            if node_id in seen_node_ids:
                raise ValueError("client phase contains a duplicate node reply")
            metric_records = list(message.content.metric_records.values())
            if len(metric_records) != 1:
                raise ValueError("client reply must contain exactly one MetricRecord")
            if message.content.config_records:
                raise ValueError("client reply must not contain a ConfigRecord")
            metric_values = dict(metric_records[0])
            if "client-id" not in metric_values:
                raise ValueError("client reply is missing client-id")
            client_id = int(metric_values.pop("client-id"))
            if client_id in seen_client_ids:
                raise ValueError("client phase contains a duplicate client-id")
            if not 0 <= client_id < int(self.min_available_nodes):
                raise ValueError("client-id is outside the configured federation")
            num_examples = int(metric_values.get(self.weighted_by_key, 0))
            if num_examples <= 0:
                raise ValueError("client reply num-examples must be positive")
            _validate_finite_metrics(metric_values)
            sent = sent_for_round.get(
                node_id,
                {
                    "payload_download_bytes": 0,
                    "model_download_bytes": 0,
                    "array_schema": (),
                },
            )
            payload_upload_bytes = recorddict_payload_bytes(message.content)
            model_upload_bytes = recorddict_array_bytes(message.content)
            if phase == "train":
                if _array_schema(message.content) != sent["array_schema"]:
                    raise ValueError("client model update schema does not match global model")
                _validate_finite_arrays(message.content)
            elif model_upload_bytes:
                raise ValueError("evaluation reply must not upload model arrays")
            if payload_upload_bytes > model_upload_bytes + MAX_CLIENT_METADATA_BYTES:
                raise ValueError("client metadata payload exceeds the configured limit")
            seen_node_ids.add(node_id)
            seen_client_ids.add(client_id)
            totals["payload_upload_bytes"] += payload_upload_bytes
            totals["model_upload_bytes"] += model_upload_bytes
            self.client_history.append(
                {
                    "round": server_round,
                    "phase": phase,
                    "client_id": client_id,
                    "node_id": node_id,
                    "num_examples": num_examples,
                    "metrics": metric_values,
                    "payload_download_bytes": sent["payload_download_bytes"],
                    "payload_upload_bytes": payload_upload_bytes,
                    "model_download_bytes": sent["model_download_bytes"],
                    "model_upload_bytes": model_upload_bytes,
                }
            )
        missing_node_ids = set(sent_for_round) - seen_node_ids
        if missing_node_ids:
            raise RuntimeError(
                f"incomplete {phase} replies for round {server_round}: "
                f"received {len(seen_node_ids)}/{len(sent_for_round)} valid replies"
            )
        return totals

    def _phase_elapsed(self, server_round: int, phase: str) -> float:
        started = self._phase_started.pop((server_round, phase), None)
        return time.perf_counter() - started if started is not None else 0.0


class TrackedFedAvg(_TrackingMixin, FedAvg):
    """FedAvg with passive evidence collection."""


class TrackedFedProx(_TrackingMixin, FedProx):
    """FedProx with passive evidence collection."""


def _array_schema(content: RecordDict) -> tuple[tuple[str, tuple[object, ...]], ...]:
    return tuple(
        (
            record_name,
            tuple(
                (array_name, array.dtype, tuple(array.shape))
                for array_name, array in record.items()
            ),
        )
        for record_name, record in content.array_records.items()
    )


def _validate_finite_arrays(content: RecordDict) -> None:
    for record in content.array_records.values():
        for array in record.values():
            if not np.isfinite(array.numpy()).all():
                raise ValueError("client model update contains NaN or Inf")


def _validate_finite_metrics(metrics: dict[str, Any]) -> None:
    for value in metrics.values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, float) and not isfinite(item):
                raise ValueError("client metrics contain NaN or Inf")


def _add_communication_metrics(
    metrics: MetricRecord,
    communication: dict[str, int],
) -> None:
    for key, value in communication.items():
        metrics[f"comm_{key}"] = int(value)
