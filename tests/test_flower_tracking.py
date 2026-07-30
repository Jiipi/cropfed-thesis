import importlib.util
import unittest

import numpy as np

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class FlowerTrackingTests(unittest.TestCase):
    def test_train_payloads_and_client_metrics_are_recorded(self) -> None:
        from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

        from cropfed.flower.tracking import (
            TrackedFedAvg,
            recorddict_payload_bytes,
        )

        strategy = TrackedFedAvg()
        outgoing = [
            Message(
                dst_node_id=node_id,
                message_type="train",
                group_id="1",
                content=RecordDict(
                    {
                        "arrays": ArrayRecord([np.asarray([1.0, 2.0])]),
                        "config": ConfigRecord({"server-round": 1}),
                    }
                ),
            )
            for node_id in (101, 202)
        ]
        strategy._track_sent(1, "train", outgoing)
        replies = [
            Message(
                content=RecordDict(
                    {
                        "arrays": ArrayRecord(
                            [np.asarray([2.0 + client_id, 3.0])]
                        ),
                        "metrics": MetricRecord(
                            {
                                "client-id": client_id,
                                "num-examples": 2 + client_id,
                                "train_loss": 0.5 + client_id,
                            }
                        ),
                    }
                ),
                reply_to=outgoing[client_id],
            )
            for client_id in (0, 1)
        ]
        expected_download = sum(
            recorddict_payload_bytes(message.content) for message in outgoing
        )
        expected_upload = sum(
            recorddict_payload_bytes(message.content) for message in replies
        )

        _, metrics = strategy.aggregate_train(1, replies)

        assert metrics is not None
        self.assertEqual(metrics["comm_payload_download_bytes"], expected_download)
        self.assertEqual(metrics["comm_payload_upload_bytes"], expected_upload)
        self.assertNotIn("client-id", metrics)
        self.assertEqual(
            [item["client_id"] for item in strategy.client_history],
            [0, 1],
        )
        self.assertEqual(strategy.client_history[0]["phase"], "train")
        self.assertGreater(strategy.client_history[0]["model_upload_bytes"], 0)

    def test_evaluate_tracks_metric_upload_without_model_upload(self) -> None:
        from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

        from cropfed.flower.tracking import TrackedFedProx

        strategy = TrackedFedProx(proximal_mu=0.01, min_available_nodes=4)
        outgoing = Message(
            dst_node_id=303,
            message_type="evaluate",
            group_id="2",
            content=RecordDict(
                {
                    "arrays": ArrayRecord([np.asarray([1.0])]),
                    "config": ConfigRecord({"server-round": 2}),
                }
            ),
        )
        strategy._track_sent(2, "evaluate", [outgoing])
        reply = Message(
            content=RecordDict(
                {
                    "metrics": MetricRecord(
                        {
                            "client-id": 3,
                            "num-examples": 4,
                            "eval_macro_f1": 0.75,
                        }
                    )
                }
            ),
            reply_to=outgoing,
        )

        metrics = strategy.aggregate_evaluate(2, [reply])

        assert metrics is not None
        self.assertEqual(metrics["comm_model_upload_bytes"], 0)
        self.assertGreater(metrics["comm_model_download_bytes"], 0)
        self.assertGreater(metrics["comm_payload_upload_bytes"], 0)
        self.assertEqual(strategy.client_history[0]["client_id"], 3)

    def test_train_rejects_non_finite_or_wrong_shape_update(self) -> None:
        from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

        from cropfed.flower.tracking import TrackedFedAvg

        for bad_array, message in (
            (np.asarray([np.nan, 2.0]), "NaN or Inf"),
            (np.asarray([1.0, 2.0, 3.0]), "schema"),
        ):
            with self.subTest(reason=message):
                strategy = TrackedFedAvg(min_available_nodes=4)
                outgoing = Message(
                    dst_node_id=101,
                    message_type="train",
                    group_id="1",
                    content=RecordDict(
                        {"arrays": ArrayRecord([np.asarray([1.0, 2.0])])}
                    ),
                )
                strategy._track_sent(1, "train", [outgoing])
                reply = Message(
                    content=RecordDict(
                        {
                            "arrays": ArrayRecord([bad_array]),
                            "metrics": MetricRecord(
                                {"client-id": 0, "num-examples": 2}
                            ),
                        }
                    ),
                    reply_to=outgoing,
                )

                with self.assertRaisesRegex(ValueError, message):
                    strategy.aggregate_train(1, [reply])

    def test_train_rejects_zero_sample_reply(self) -> None:
        from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

        from cropfed.flower.tracking import TrackedFedAvg

        strategy = TrackedFedAvg(min_available_nodes=4)
        outgoing = Message(
            dst_node_id=101,
            message_type="train",
            group_id="1",
            content=RecordDict({"arrays": ArrayRecord([np.asarray([1.0])])}),
        )
        strategy._track_sent(1, "train", [outgoing])
        reply = Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord([np.asarray([2.0])]),
                    "metrics": MetricRecord({"client-id": 0, "num-examples": 0}),
                }
            ),
            reply_to=outgoing,
        )

        with self.assertRaisesRegex(ValueError, "positive"):
            strategy.aggregate_train(1, [reply])

    def test_train_rejects_incomplete_valid_reply_set(self) -> None:
        from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

        from cropfed.flower.tracking import TrackedFedAvg

        strategy = TrackedFedAvg()
        outgoing = [
            Message(
                dst_node_id=node_id,
                message_type="train",
                group_id="1",
                content=RecordDict(
                    {
                        "arrays": ArrayRecord([np.asarray([1.0])]),
                        "config": ConfigRecord({"server-round": 1}),
                    }
                ),
            )
            for node_id in (101, 202)
        ]
        strategy._track_sent(1, "train", outgoing)
        only_reply = Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord([np.asarray([2.0])]),
                    "metrics": MetricRecord(
                        {"client-id": 0, "num-examples": 2}
                    ),
                }
            ),
            reply_to=outgoing[0],
        )

        with self.assertRaisesRegex(RuntimeError, "received 1/2 valid replies"):
            strategy.aggregate_train(1, [only_reply])


if __name__ == "__main__":
    unittest.main()
