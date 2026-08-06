"""Flower strategies with auditable per-client metrics and payload byte counts."""

from __future__ import annotations

import time
from collections.abc import Iterable
from math import isfinite
from typing import Any

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.common import MessageType
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.serverapp.strategy.strategy_utils import sample_nodes

MAX_CLIENT_METADATA_BYTES = 1_000_000

#: Server → client: the SCAFFOLD server control variate ``c``.
SCAFFOLD_C_RECORD = "scaffold_c"
#: Client → server: the per-client control variate delta ``c_i⁺ - c_i``.
SCAFFOLD_C_DELTA_RECORD = "scaffold_c_delta"


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

    #: Array records a client may return in addition to the model update.
    #: SCAFFOLD adds a control-variate delta here; these records are excluded
    #: from the model-schema equality check but still counted as real bytes on
    #: the wire, because they genuinely are transmitted.
    auxiliary_array_records: frozenset[str] = frozenset()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.client_history: list[dict[str, Any]] = []
        self._sent_messages: dict[tuple[int, str, int], dict[str, Any]] = {}
        self._phase_started: dict[tuple[int, str], float] = {}
        self._pending_round_state: dict[int, dict[str, Any]] = {}
        self.best_validation_round: int | None = None
        self.best_validation_metrics: dict[str, Any] | None = None
        self.best_state_dict: dict[str, Any] | None = None

    def _model_schema(self, content: RecordDict):
        """Schema of the model update only, ignoring auxiliary records."""

        return tuple(
            item
            for item in _array_schema(content)
            if item[0] not in self.auxiliary_array_records
        )

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
        # Auxiliary records must be removed before the parent aggregation runs:
        # Flower merges every ArrayRecord in a reply by key, and a SCAFFOLD
        # control-variate delta shares its keys with the model parameters, so
        # leaving it in place would add the delta straight into the weights.
        auxiliary = self._take_auxiliary_arrays(reply_list)
        arrays, metrics = super().aggregate_train(server_round, reply_list)
        if arrays is not None:
            import torch

            self._pending_round_state[server_round] = {
                name: torch.clone(tensor.detach().cpu())
                for name, tensor in arrays.to_torch_state_dict().items()
            }
        if metrics is not None:
            metrics.pop("client-id", None)
            _add_communication_metrics(metrics, communication)
            metrics["phase_seconds"] = self._phase_elapsed(server_round, "train")
        if self.auxiliary_array_records:
            # Called even when nothing was collected: an empty set of auxiliary
            # records is precisely the silent-degradation case a strategy needs
            # to reject.
            arrays, metrics = self.on_auxiliary_arrays(
                server_round, auxiliary, arrays, metrics
            )
        return arrays, metrics

    def _take_auxiliary_arrays(
        self,
        replies: list[Message],
    ) -> dict[str, list[dict[str, Any]]]:
        """Remove and return auxiliary array records, keyed by record name."""

        collected: dict[str, list[dict[str, Any]]] = {}
        if not self.auxiliary_array_records:
            return collected
        for message in replies:
            if message.has_error() or not message.has_content():
                continue
            for record_name in self.auxiliary_array_records:
                if record_name not in message.content.array_records:
                    continue
                record = message.content.pop(record_name)
                collected.setdefault(record_name, []).append(
                    {
                        name: array.numpy()
                        for name, array in record.items()
                    }
                )
        return collected

    def on_auxiliary_arrays(
        self,
        server_round: int,
        auxiliary: dict[str, list[dict[str, Any]]],
        arrays,
        metrics,
    ):
        """Hook for strategies that carry extra client state. Default: ignore."""

        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        reply_list = list(replies)
        communication = self._track_replies(server_round, "evaluate", reply_list)
        metrics = super().aggregate_evaluate(server_round, reply_list)
        if metrics is not None:
            metrics.pop("client-id", None)
            _add_communication_metrics(metrics, communication)
            metrics["phase_seconds"] = self._phase_elapsed(server_round, "evaluate")
            self._consider_validation_checkpoint(server_round, metrics)
        else:
            self._pending_round_state.pop(server_round, None)
        return metrics

    def _consider_validation_checkpoint(
        self,
        server_round: int,
        metrics: MetricRecord,
    ) -> None:
        state_dict = self._pending_round_state.pop(server_round, None)
        if state_dict is None or "eval_macro_f1" not in metrics:
            return
        candidate_f1 = float(metrics["eval_macro_f1"])
        candidate_loss = float(metrics.get("eval_loss", float("inf")))
        incumbent = self.best_validation_metrics
        is_better = incumbent is None
        if incumbent is not None:
            incumbent_f1 = float(incumbent["eval_macro_f1"])
            incumbent_loss = float(incumbent.get("eval_loss", float("inf")))
            is_better = candidate_f1 > incumbent_f1 or (
                candidate_f1 == incumbent_f1 and candidate_loss < incumbent_loss
            )
        if is_better:
            self.best_validation_round = server_round
            self.best_validation_metrics = dict(metrics)
            self.best_state_dict = state_dict

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
                "array_schema": self._model_schema(message.content),
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
                if self._model_schema(message.content) != sent["array_schema"]:
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


class TrackedFedBN(_TrackingMixin, FedAvg):
    """FedBN (Li et al., 2021): batch-norm statistics stay on the client.

    Server aggregation is FedAvg over every tensor, because the exclusion is
    enforced where it belongs — the client restores its own BN statistics over
    the averaged ones before training and before validating.  This strategy's
    job is therefore to prove the clients really did so: from round 2 onwards
    every client must report a non-zero ``fedbn_local_bn_tensors``, otherwise
    the run is plain FedAvg wearing a FedBN label.

    Round 1 is exempt: no client has local statistics to restore yet.
    """

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        reply_list = list(replies)
        arrays, metrics = super().aggregate_train(server_round, reply_list)
        if metrics is not None and server_round > 1:
            restored = float(metrics.get("fedbn_local_bn_tensors", 0.0))
            if restored <= 0.0:
                raise RuntimeError(
                    "FedBN clients restored no local batch-norm statistics in "
                    f"round {server_round}; local training used the averaged BN "
                    "and the result would be plain FedAvg"
                )
        return arrays, metrics


class TrackedSCAFFOLD(_TrackingMixin, FedAvg):
    """SCAFFOLD (Karimireddy et al., 2020) over the Flower message API.

    Each round the server ships its control variate ``c`` alongside the global
    weights, and each client returns ``c_i⁺ - c_i`` in a separate ArrayRecord.
    The server then applies ``c ← c + (1/N) · Σᵢ (c_i⁺ - c_i)``, where N is the
    number of participating clients — the standard SCAFFOLD (option II) update.

    ``scaffold_server_lr`` scales the aggregated weight update, which is the
    global step size ``η_g`` in the paper.
    """

    auxiliary_array_records = frozenset({SCAFFOLD_C_DELTA_RECORD})

    def __init__(self, *args, scaffold_server_lr: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if scaffold_server_lr <= 0:
            raise ValueError("scaffold_server_lr must be positive")
        self.scaffold_server_lr = scaffold_server_lr
        self._scaffold_c: dict[str, np.ndarray] | None = None
        #: Weights sent this round, needed to apply the global step size.
        self._round_arrays: dict[str, np.ndarray] | None = None
        self.rounds_with_control_variates: int = 0

    def configure_train(self, server_round, arrays, config, grid):
        """Attach the server control variate ``c`` to the training message."""

        if self._scaffold_c is None:
            self._scaffold_c = {
                name: np.zeros(array.numpy().shape, dtype=np.float64)
                for name, array in arrays.items()
            }
        self._round_arrays = {
            name: array.numpy().copy() for name, array in arrays.items()
        }
        self._phase_started[(server_round, "train")] = time.perf_counter()
        config["scaffold-server-lr"] = float(self.scaffold_server_lr)
        config["server-round"] = server_round
        record = RecordDict(
            {
                self.arrayrecord_key: arrays,
                self.configrecord_key: config,
                SCAFFOLD_C_RECORD: _array_record_from_numpy(self._scaffold_c),
            }
        )
        node_ids, _ = _sample_train_nodes(self, grid)
        messages = list(
            self._construct_messages(record, node_ids, MessageType.TRAIN)
        )
        self._track_sent(server_round, "train", messages)
        return messages

    def on_auxiliary_arrays(self, server_round, auxiliary, arrays, metrics):
        """Apply ``c ← c + (1/N)·Σ Δc_i`` and the global step size ``η_g``."""

        deltas = auxiliary.get(SCAFFOLD_C_DELTA_RECORD, [])
        if not deltas:
            raise RuntimeError(
                "SCAFFOLD received no client control-variate deltas in round "
                f"{server_round}; the clients are not running SCAFFOLD and the "
                "result would silently be plain FedAvg"
            )
        assert self._scaffold_c is not None
        for name in self._scaffold_c:
            present = [delta[name] for delta in deltas if name in delta]
            if not present:
                continue
            mean_delta = np.mean(
                np.stack([value.astype(np.float64) for value in present]), axis=0
            )
            self._scaffold_c[name] = self._scaffold_c[name] + mean_delta
        self.rounds_with_control_variates += 1

        # Global step size: w ← w_prev + η_g · (w_agg - w_prev)
        if arrays is not None and self.scaffold_server_lr != 1.0:
            assert self._round_arrays is not None
            import torch

            stepped = {}
            for name, tensor in arrays.to_torch_state_dict().items():
                aggregated = tensor.detach().cpu().numpy().astype(np.float64)
                previous = self._round_arrays.get(name)
                if previous is None:
                    stepped[name] = tensor.detach().cpu()
                    continue
                updated = previous + self.scaffold_server_lr * (
                    aggregated - previous
                )
                stepped[name] = torch.as_tensor(
                    updated.astype(previous.dtype), dtype=tensor.dtype
                )
            arrays = ArrayRecord(stepped)
            self._pending_round_state[server_round] = {
                name: value.clone() for name, value in stepped.items()
            }
        if metrics is not None:
            metrics["scaffold_clients_reporting"] = len(deltas)
        return arrays, metrics


class TrackedMOON(_TrackingMixin, FedAvg):
    """MOON (Li et al., 2021): model-contrastive local training.

    Server aggregation is plain FedAvg; the contribution is on the client, so
    this strategy's job is to ship ``μ`` and ``τ`` and to verify that clients
    actually applied the contrastive loss.
    """

    def __init__(self, *args, moon_temperature: float = 0.5, moon_mu: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if moon_temperature <= 0:
            raise ValueError("moon_temperature must be positive")
        if moon_mu < 0:
            raise ValueError("moon_mu cannot be negative")
        self.moon_temperature = moon_temperature
        self.moon_mu = moon_mu

    def configure_train(self, server_round, arrays, config, grid):
        """Inject the MOON hyperparameters clients need to build the loss."""

        config["moon-temperature"] = float(self.moon_temperature)
        config["moon-mu"] = float(self.moon_mu)
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        reply_list = list(replies)
        arrays, metrics = super().aggregate_train(server_round, reply_list)
        # Round 1 has no previous local model, so MOON legitimately reports no
        # contrastive term; from round 2 onwards a silent absence would mean the
        # clients ran plain FedAvg.
        if metrics is not None and server_round > 1 and self.moon_mu > 0:
            if "moon_contrastive_loss" not in metrics:
                raise RuntimeError(
                    "MOON clients did not report 'moon_contrastive_loss' in "
                    f"round {server_round}; local training did not apply the "
                    "contrastive term and the result would be plain FedAvg"
                )
        return arrays, metrics


def _array_record_from_numpy(values: dict[str, np.ndarray]) -> ArrayRecord:
    """Build an ArrayRecord from named numpy arrays.

    ``ArrayRecord`` accepts a list of ndarrays, a torch state dict, or a dict of
    ``Array`` — but not a dict of raw ndarrays, which would silently be taken as
    a positional argument and rejected.
    """

    return ArrayRecord(
        {
            name: Array(np.ascontiguousarray(value, dtype=np.float32))
            for name, value in values.items()
        }
    )


def _sample_train_nodes(strategy, grid) -> tuple[list[int], list[int]]:
    """Sample training nodes exactly as ``FedAvg.configure_train`` does."""

    num_nodes = int(len(list(grid.get_node_ids())) * strategy.fraction_train)
    sample_size = max(num_nodes, strategy.min_train_nodes)
    return sample_nodes(grid, strategy.min_available_nodes, sample_size)


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
