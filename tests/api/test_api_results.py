import unittest

from cropfed.api.results import summarize_round


class ExperimentResultNormalizationTests(unittest.TestCase):
    def test_synthetic_round_columns_are_extracted(self) -> None:
        summary = summarize_round(
            {
                "round": 2,
                "train_loss": 0.9,
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "round_seconds": 1.25,
                "bytes_up": 100,
                "bytes_down": 120,
            }
        )

        self.assertEqual(summary["train_loss"], 0.9)
        self.assertEqual(summary["accuracy"], 0.8)
        self.assertEqual(summary["macro_f1"], 0.7)
        self.assertEqual(summary["elapsed_seconds"], 1.25)
        self.assertEqual(summary["bytes_up"], 100)
        self.assertEqual(summary["bytes_down"], 120)

    def test_flower_central_metrics_take_precedence(self) -> None:
        summary = summarize_round(
            {
                "round": 1,
                "train": {"train_loss": 1.2},
                "federated_evaluate": {
                    "eval_accuracy": 0.4,
                    "eval_macro_f1": 0.3,
                },
                "central_evaluate": {
                    "central_loss": 0.6,
                    "central_accuracy": 0.8,
                    "central_macro_f1": 0.75,
                    "central_harmful_missed_as_healthy_rate": 0.1,
                },
            }
        )

        self.assertEqual(summary["train_loss"], 1.2)
        self.assertEqual(summary["evaluation_loss"], 0.6)
        self.assertEqual(summary["accuracy"], 0.8)
        self.assertEqual(summary["macro_f1"], 0.75)
        self.assertEqual(summary["harmful_missed_as_healthy_rate"], 0.1)

    def test_missing_metrics_remain_null(self) -> None:
        summary = summarize_round({"round": 0, "central_evaluate": {}})

        self.assertIsNone(summary["accuracy"])
        self.assertIsNone(summary["macro_f1"])
        self.assertIsNone(summary["bytes_up"])


if __name__ == "__main__":
    unittest.main()
