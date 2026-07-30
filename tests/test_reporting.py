import unittest
from types import SimpleNamespace

from cropfed.constants import TOMATO_CLASSES
from cropfed.ml.metrics import classification_metrics
from cropfed.ml.reporting import flower_evaluation_values


class FlowerReportingTests(unittest.TestCase):
    def test_detailed_values_preserve_class_order_and_confusion_matrix(self) -> None:
        targets = list(range(10))
        predictions = [0, 0, *range(2, 10)]
        metrics = classification_metrics(
            targets,
            predictions,
            num_classes=10,
            class_names=TOMATO_CLASSES,
            healthy_class_id=0,
            class_groups=[
                "healthy",
                "disease",
                "disease",
                "disease",
                "disease",
                "disease",
                "disease",
                "disease",
                "disease",
                "pest",
            ],
        )
        evaluation = SimpleNamespace(loss=1.25, metrics=metrics)

        values = flower_evaluation_values(
            evaluation,
            prefix="central",
            detailed=True,
        )

        self.assertEqual(values["central_harmful_missed_as_healthy_count"], 1)
        self.assertEqual(len(values["central_per_class_recall"]), 10)
        self.assertEqual(len(values["central_per_class_precision"]), 10)
        self.assertEqual(len(values["central_confusion_matrix_flat"]), 100)
        self.assertEqual(values["central_confusion_matrix_size"], 10)
        self.assertEqual(values["central_spider_mite_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
