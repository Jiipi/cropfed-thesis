import unittest

from cropfed.ml.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_known_multiclass_values(self) -> None:
        metrics = classification_metrics(
            y_true=[0, 0, 1, 1, 2, 2],
            y_pred=[0, 1, 1, 1, 2, 0],
            num_classes=3,
        )
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(metrics["per_class"]["0"]["support"], 2)
        self.assertEqual(len(metrics["confusion_matrix"]), 3)

    def test_zero_prediction_class_does_not_divide_by_zero(self) -> None:
        metrics = classification_metrics([0, 1], [0, 0], num_classes=2)
        self.assertEqual(metrics["per_class"]["1"]["precision"], 0.0)
        self.assertEqual(metrics["per_class"]["1"]["f1"], 0.0)

    def test_harmful_as_healthy_and_group_metrics(self) -> None:
        metrics = classification_metrics(
            y_true=[0, 1, 2, 2],
            y_pred=[0, 0, 1, 2],
            num_classes=3,
            class_names=["healthy", "disease-a", "pest-a"],
            healthy_class_id=0,
            class_groups=["healthy", "disease", "pest"],
        )

        self.assertEqual(metrics["harmful_missed_as_healthy_count"], 1)
        self.assertAlmostEqual(metrics["harmful_missed_as_healthy_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["harmful_detection_recall"], 2 / 3)
        self.assertIn("disease", metrics["group_metrics"]["per_class"])


if __name__ == "__main__":
    unittest.main()
