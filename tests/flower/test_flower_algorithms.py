"""SCAFFOLD and MOON must not silently degrade into FedAvg.

The failure mode these tests exist for: the server aggregates with the right
formula while the clients never apply the algorithm, producing a results table
that looks valid and differs only by noise. Every check here is about state
actually crossing the wire, or a loud failure when it does not.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


def _numpy_record(values: dict):
    """ArrayRecord rejects a dict of raw ndarrays; wrap them in Array."""

    from flwr.app import Array, ArrayRecord

    return ArrayRecord(
        {
            name: Array(np.ascontiguousarray(value, dtype=np.float32))
            for name, value in values.items()
        }
    )


def _train_message(node_id: int, arrays, extra=None):
    from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict

    records = {
        "arrays": ArrayRecord([np.asarray(value) for value in arrays]),
        "config": ConfigRecord({"server-round": 1}),
    }
    if extra:
        records.update(extra)
    return Message(
        dst_node_id=node_id,
        message_type="train",
        group_id="1",
        content=RecordDict(records),
    )


def _train_reply(outgoing, client_id: int, arrays, extra=None, metrics=None):
    from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

    values = {"client-id": client_id, "num-examples": 4}
    if metrics:
        values.update(metrics)
    records = {
        "arrays": ArrayRecord([np.asarray(value) for value in arrays]),
        "metrics": MetricRecord(values),
    }
    if extra:
        records.update(extra)
    return Message(content=RecordDict(records), reply_to=outgoing)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class ScaffoldStrategyTests(unittest.TestCase):
    def test_control_variate_delta_is_not_averaged_into_the_model(self) -> None:
        """The delta record must be stripped before FedAvg aggregation.

        Flower merges every ArrayRecord in a reply by key. Because the SCAFFOLD
        delta shares parameter names with the model, leaving it in place adds
        the control variate straight into the weights.
        """

        from cropfed.flower.tracking import SCAFFOLD_C_DELTA_RECORD, TrackedSCAFFOLD

        strategy = TrackedSCAFFOLD(min_available_nodes=2)
        outgoing = [_train_message(node_id, [[0.0, 0.0]]) for node_id in (101, 202)]
        strategy._track_sent(1, "train", outgoing)
        strategy._scaffold_c = {"0": np.zeros(2)}

        replies = [
            _train_reply(
                outgoing[client_id],
                client_id,
                [[2.0, 4.0]],
                extra={
                    SCAFFOLD_C_DELTA_RECORD: _numpy_record(
                        {"0": np.asarray([100.0, 100.0])}
                    )
                },
            )
            for client_id in (0, 1)
        ]

        arrays, metrics = strategy.aggregate_train(1, replies)

        assert arrays is not None
        aggregated = next(iter(arrays.to_torch_state_dict().values())).numpy()
        np.testing.assert_allclose(aggregated, np.asarray([2.0, 4.0]), atol=1e-5)
        assert metrics is not None
        self.assertEqual(metrics["scaffold_clients_reporting"], 2)

    def test_server_control_variate_accumulates_client_deltas(self) -> None:
        """c ← c + mean(Δc_i), the SCAFFOLD option-II server update."""

        from cropfed.flower.tracking import SCAFFOLD_C_DELTA_RECORD, TrackedSCAFFOLD

        strategy = TrackedSCAFFOLD(min_available_nodes=2)
        outgoing = [_train_message(node_id, [[0.0, 0.0]]) for node_id in (101, 202)]
        strategy._track_sent(1, "train", outgoing)
        strategy._scaffold_c = {"0": np.zeros(2)}

        replies = [
            _train_reply(
                outgoing[client_id],
                client_id,
                [[1.0, 1.0]],
                extra={
                    SCAFFOLD_C_DELTA_RECORD: _numpy_record({"0": np.asarray(delta)})
                },
            )
            for client_id, delta in enumerate([[2.0, 4.0], [4.0, 8.0]])
        ]
        strategy.aggregate_train(1, replies)

        assert strategy._scaffold_c is not None
        np.testing.assert_allclose(
            strategy._scaffold_c["0"], np.asarray([3.0, 6.0]), atol=1e-5
        )
        self.assertEqual(strategy.rounds_with_control_variates, 1)

    def test_missing_control_variates_raise_instead_of_running_as_fedavg(self) -> None:
        from cropfed.flower.tracking import TrackedSCAFFOLD

        strategy = TrackedSCAFFOLD(min_available_nodes=2)
        outgoing = [_train_message(node_id, [[0.0, 0.0]]) for node_id in (101, 202)]
        strategy._track_sent(1, "train", outgoing)
        strategy._scaffold_c = {"0": np.zeros(2)}

        replies = [
            _train_reply(outgoing[client_id], client_id, [[1.0, 1.0]])
            for client_id in (0, 1)
        ]

        with self.assertRaisesRegex(RuntimeError, "silently be plain FedAvg"):
            strategy.aggregate_train(1, replies)

    def test_rejects_non_positive_server_learning_rate(self) -> None:
        from cropfed.flower.tracking import TrackedSCAFFOLD

        with self.assertRaisesRegex(ValueError, "scaffold_server_lr"):
            TrackedSCAFFOLD(scaffold_server_lr=0.0)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class MoonStrategyTests(unittest.TestCase):
    def test_hyperparameters_are_injected_into_the_train_config(self) -> None:
        """Clients cannot build the contrastive loss without μ and τ.

        ``configure_train`` blocks on real node sampling, so the injection is
        exercised through a stubbed parent rather than a live Grid.
        """

        from flwr.app import ConfigRecord

        from cropfed.flower.tracking import TrackedMOON

        seen: dict[str, object] = {}

        class _CapturingMOON(TrackedMOON):
            def _construct_messages(self, record, node_ids, message_type):
                seen["config"] = dict(record["config"])
                return []

            def _sampled(self, grid):
                return [], []

        strategy = _CapturingMOON(moon_temperature=0.7, moon_mu=2.0)
        config = ConfigRecord({})

        class _Grid:
            def get_node_ids(self):
                return [1, 2]

        strategy.fraction_train = 0.0  # short-circuits sampling in FedAvg
        strategy.configure_train(1, None, config, _Grid())

        self.assertEqual(config["moon-temperature"], 0.7)
        self.assertEqual(config["moon-mu"], 2.0)

    def test_absent_contrastive_metric_raises_after_the_first_round(self) -> None:
        from cropfed.flower.tracking import TrackedMOON

        strategy = TrackedMOON(min_available_nodes=2)
        outgoing = [_train_message(node_id, [[0.0]]) for node_id in (101, 202)]
        strategy._track_sent(2, "train", outgoing)
        replies = [
            _train_reply(outgoing[client_id], client_id, [[1.0]])
            for client_id in (0, 1)
        ]

        with self.assertRaisesRegex(RuntimeError, "plain FedAvg"):
            strategy.aggregate_train(2, replies)

    def test_first_round_tolerates_absent_contrastive_metric(self) -> None:
        """Round 1 has no previous local model, so MOON is legitimately inert."""

        from cropfed.flower.tracking import TrackedMOON

        strategy = TrackedMOON(min_available_nodes=2)
        outgoing = [_train_message(node_id, [[0.0]]) for node_id in (101, 202)]
        strategy._track_sent(1, "train", outgoing)
        replies = [
            _train_reply(outgoing[client_id], client_id, [[1.0]])
            for client_id in (0, 1)
        ]

        _, metrics = strategy.aggregate_train(1, replies)
        self.assertIsNotNone(metrics)

    def test_reported_contrastive_loss_is_accepted(self) -> None:
        from cropfed.flower.tracking import TrackedMOON

        strategy = TrackedMOON(min_available_nodes=2)
        outgoing = [_train_message(node_id, [[0.0]]) for node_id in (101, 202)]
        strategy._track_sent(2, "train", outgoing)
        replies = [
            _train_reply(
                outgoing[client_id],
                client_id,
                [[1.0]],
                metrics={"moon_contrastive_loss": 0.5},
            )
            for client_id in (0, 1)
        ]

        _, metrics = strategy.aggregate_train(2, replies)
        assert metrics is not None
        self.assertAlmostEqual(float(metrics["moon_contrastive_loss"]), 0.5)

    def test_rejects_invalid_hyperparameters(self) -> None:
        from cropfed.flower.tracking import TrackedMOON

        with self.assertRaisesRegex(ValueError, "moon_temperature"):
            TrackedMOON(moon_temperature=0.0)
        with self.assertRaisesRegex(ValueError, "moon_mu"):
            TrackedMOON(moon_mu=-1.0)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class ClientAlgorithmStateTests(unittest.TestCase):
    """The client half of the contract: consume server state, refuse to guess."""

    def _context(self, algorithm: str):
        from flwr.app import Context, RecordDict

        return Context(
            run_id=1,
            node_id=1,
            node_config={"partition-id": 0},
            state=RecordDict(),
            run_config={
                "algorithm": algorithm,
                "model-name": "mobilenet_v3_small",
                "taxonomy-scope": "plantvillage-full",
            },
        )

    def test_scaffold_client_raises_when_server_sends_no_control_variate(self) -> None:
        from cropfed.flower.client_app import _scaffold_train_kwargs

        message = _train_message(101, [[0.0]])
        with self.assertRaisesRegex(RuntimeError, "silently run"):
            _scaffold_train_kwargs(message, self._context("scaffold"), None, "scaffold")

    def test_moon_client_raises_when_server_sends_no_hyperparameters(self) -> None:
        from cropfed.flower.client_app import (
            MOON_PREVIOUS_MODEL_RECORD,
            _moon_train_kwargs,
        )

        context = self._context("moon")
        # A previous model exists, so round-1 leniency does not apply.
        context.state[MOON_PREVIOUS_MODEL_RECORD] = _numpy_record(
            {"0": np.asarray([1.0])}
        )
        message = _train_message(101, [[0.0]])

        with self.assertRaisesRegex(RuntimeError, "silently run as FedAvg"):
            _moon_train_kwargs(message, context, None, {}, "moon")

    def test_moon_first_round_returns_no_contrastive_references(self) -> None:
        """Round 1 has no previous local model, so the loss is undefined."""

        from flwr.app import ConfigRecord

        from cropfed.flower.client_app import _moon_train_kwargs

        message = _train_message(101, [[0.0]])
        message.content["config"] = ConfigRecord(
            {"server-round": 1, "moon-mu": 1.0, "moon-temperature": 0.5}
        )
        self.assertEqual(
            _moon_train_kwargs(message, self._context("moon"), None, {}, "moon"),
            {},
        )

    def test_other_algorithms_add_no_algorithm_state(self) -> None:
        from cropfed.flower.client_app import _moon_train_kwargs, _scaffold_train_kwargs

        message = _train_message(101, [[0.0]])
        for algorithm in ("fedavg", "fedprox", "fedbn"):
            with self.subTest(algorithm=algorithm):
                context = self._context(algorithm)
                self.assertEqual(
                    _scaffold_train_kwargs(message, context, None, algorithm), {}
                )
                self.assertEqual(
                    _moon_train_kwargs(message, context, None, {}, algorithm), {}
                )

    def test_scaffold_previous_variate_round_trips_through_client_state(self) -> None:
        """Round N's c_i must be readable in round N+1 from the client's state."""

        import torch

        from cropfed.flower.client_app import (
            _scaffold_previous_c_i,
            _store_client_arrays,
        )
        from cropfed.flower.tracking import SCAFFOLD_C_DELTA_RECORD  # noqa: F401

        context = self._context("scaffold")

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        model = _Model()
        zeros = _scaffold_previous_c_i(context, model)
        for value in zeros.values():
            self.assertTrue(torch.all(value == 0.0))

        stored = {name: torch.full_like(value, 0.25) for name, value in zeros.items()}
        _store_client_arrays(context, "scaffold_client_c", stored)

        recovered = _scaffold_previous_c_i(context, model)
        for name, value in recovered.items():
            torch.testing.assert_close(value, stored[name])


if __name__ == "__main__":
    unittest.main()
