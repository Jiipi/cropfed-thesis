import unittest
from types import SimpleNamespace

from cropfed.constants import PLANTVILLAGE_FULL_TAXONOMY, TOMATO_CLASSES
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
            class_names=TOMATO_CLASSES,
        )

        self.assertEqual(values["central_harmful_missed_as_healthy_count"], 1)
        self.assertEqual(len(values["central_per_class_recall"]), 10)
        self.assertEqual(len(values["central_per_class_precision"]), 10)
        self.assertEqual(len(values["central_confusion_matrix_flat"]), 100)
        self.assertEqual(values["central_confusion_matrix_size"], 10)
        self.assertEqual(values["central_spider_mite_f1"], 1.0)

    def test_full_taxonomy_reports_38_classes_and_multiple_healthy(self) -> None:
        """The 38-class taxonomy has twelve healthy classes, not one.

        ``harmful_missed_as_healthy`` must treat every crop's healthy class as
        healthy, otherwise the safety metric silently under-counts misses.
        """

        taxonomy = PLANTVILLAGE_FULL_TAXONOMY
        num_classes = len(taxonomy.class_names)
        healthy_ids = taxonomy.healthy_class_ids
        targets = list(range(num_classes))
        # Send one harmful class to a healthy class to exercise the safety path.
        harmful_id = next(
            index for index in range(num_classes) if index not in healthy_ids
        )
        predictions = list(range(num_classes))
        predictions[harmful_id] = healthy_ids[0]

        metrics = classification_metrics(
            targets,
            predictions,
            num_classes=num_classes,
            class_names=taxonomy.class_names,
            healthy_class_ids=healthy_ids,
            class_groups=taxonomy.class_groups,
        )
        evaluation = SimpleNamespace(loss=0.5, metrics=metrics)

        values = flower_evaluation_values(
            evaluation,
            prefix="central",
            detailed=True,
            class_names=taxonomy.class_names,
        )

        self.assertEqual(len(healthy_ids), 12)
        self.assertEqual(values["central_confusion_matrix_size"], 38)
        self.assertEqual(len(values["central_confusion_matrix_flat"]), 38 * 38)
        self.assertEqual(len(values["central_per_class_f1"]), 38)
        self.assertEqual(values["central_harmful_missed_as_healthy_count"], 1)
        # Group metrics must resolve rather than raise for the full taxonomy.
        self.assertIn("central_disease_f1", values)
        self.assertIn("central_pest_f1", values)

    def test_missing_class_group_names_the_taxonomy_mismatch(self) -> None:
        """A taxonomy without a pest class must fail loudly, not report 0.0."""

        names = ("Healthy", "Blight")
        metrics = classification_metrics(
            [0, 1],
            [0, 1],
            num_classes=2,
            class_names=names,
            healthy_class_ids=(0,),
            class_groups=("healthy", "disease"),
        )
        evaluation = SimpleNamespace(loss=0.1, metrics=metrics)

        with self.assertRaises(KeyError) as caught:
            flower_evaluation_values(
                evaluation, prefix="central", detailed=False, class_names=names
            )

        self.assertIn("pest", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
