import importlib.util
import unittest

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class FlowerServerResultTests(unittest.TestCase):
    def test_result_history_preserves_initial_and_round_metrics(self) -> None:
        from flwr.app import MetricRecord
        from flwr.serverapp.strategy.result import Result

        from cropfed.flower.server_app import _result_history

        result = Result()
        result.train_metrics_clientapp[1] = MetricRecord({"train_loss": 2.0})
        result.evaluate_metrics_clientapp[1] = MetricRecord(
            {"eval_macro_f1": 0.4}
        )
        result.evaluate_metrics_serverapp[0] = MetricRecord(
            {"central_macro_f1": 0.1}
        )
        result.evaluate_metrics_serverapp[1] = MetricRecord(
            {"central_macro_f1": 0.5}
        )

        history = _result_history(result)

        self.assertEqual([row["round"] for row in history], [0, 1])
        self.assertEqual(history[0]["central_evaluate"]["central_macro_f1"], 0.1)
        self.assertEqual(history[1]["train"]["train_loss"], 2.0)
        self.assertEqual(
            history[1]["federated_evaluate"]["eval_macro_f1"], 0.4
        )
        self.assertEqual(history[0]["communication"]["payload_total_bytes"], 0)

    def test_result_history_sums_train_and_evaluate_communication(self) -> None:
        from flwr.app import MetricRecord
        from flwr.serverapp.strategy.result import Result

        from cropfed.flower.server_app import _communication_summary, _result_history

        result = Result()
        result.train_metrics_clientapp[1] = MetricRecord(
            {
                "comm_payload_download_bytes": 100,
                "comm_payload_upload_bytes": 80,
                "comm_model_download_bytes": 90,
                "comm_model_upload_bytes": 70,
            }
        )
        result.evaluate_metrics_clientapp[1] = MetricRecord(
            {
                "comm_payload_download_bytes": 50,
                "comm_payload_upload_bytes": 10,
                "comm_model_download_bytes": 45,
                "comm_model_upload_bytes": 0,
            }
        )

        history = _result_history(result)
        communication = history[0]["communication"]
        summary = _communication_summary(history)

        self.assertEqual(communication["payload_download_bytes"], 150)
        self.assertEqual(communication["payload_upload_bytes"], 90)
        self.assertEqual(communication["payload_total_bytes"], 240)
        self.assertEqual(summary["payload_total_bytes"], 240)


if __name__ == "__main__":
    unittest.main()
