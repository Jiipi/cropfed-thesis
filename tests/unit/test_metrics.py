import unittest

from cropfed.ml.metrics import (
    classification_metrics,
    client_fairness,
    gap_vs_centralized,
)


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


class ClientFairnessTests(unittest.TestCase):
    def test_known_spread_values(self) -> None:
        fairness = client_fairness([0.4, 0.6, 0.8, 1.0])

        self.assertEqual(fairness["num_clients"], 4)
        self.assertAlmostEqual(fairness["mean"], 0.7)
        self.assertAlmostEqual(fairness["worst"], 0.4)
        self.assertAlmostEqual(fairness["best"], 1.0)
        self.assertAlmostEqual(fairness["spread"], 0.6)
        # Population std of [0.4, 0.6, 0.8, 1.0] is sqrt(0.05) ~= 0.2236;
        # the sample std would be 0.2582, which is the wrong convention here.
        self.assertAlmostEqual(fairness["std"], 0.223606797749979)
        self.assertAlmostEqual(fairness["coefficient_of_variation"], 0.3194382825)

    def test_population_std_not_sample_std(self) -> None:
        """A four-client federation is not a sample drawn from a larger one.

        Dividing by n-1 would inflate the reported unfairness by ~15% at four
        clients, making a run look like it abandoned a client more than it did.
        """

        fairness = client_fairness([0.5, 0.9])

        self.assertAlmostEqual(fairness["std"], 0.2)
        self.assertNotAlmostEqual(fairness["std"], 0.2828427, places=3)

    def test_perfectly_equal_clients_have_zero_spread(self) -> None:
        fairness = client_fairness([0.75, 0.75, 0.75])

        self.assertEqual(fairness["std"], 0.0)
        self.assertEqual(fairness["spread"], 0.0)
        self.assertEqual(fairness["coefficient_of_variation"], 0.0)

    def test_unweighted_mean_does_not_hide_an_abandoned_small_client(self) -> None:
        """The point of the fairness metric, stated as a test.

        One tiny client scoring 0.10 against three large ones at 0.90: the
        weighted mean stays at 0.88 and looks healthy, while the unweighted mean
        drops and ``size_advantage`` names the effect explicitly.
        """

        fairness = client_fairness(
            [0.10, 0.90, 0.90, 0.90],
            num_examples=[10, 1_000, 1_000, 1_000],
        )

        self.assertAlmostEqual(fairness["mean"], 0.70)
        self.assertGreater(fairness["weighted_mean"], 0.88)
        self.assertGreater(fairness["size_advantage"], 0.18)
        self.assertAlmostEqual(fairness["smallest_client_score"], 0.10)
        self.assertEqual(fairness["smallest_client_examples"], 10)
        self.assertAlmostEqual(fairness["largest_client_score"], 0.90)

    def test_no_size_advantage_when_scores_do_not_track_size(self) -> None:
        fairness = client_fairness([0.8, 0.8, 0.8], num_examples=[10, 100, 1_000])

        self.assertAlmostEqual(fairness["size_advantage"], 0.0)

    def test_zero_mean_leaves_coefficient_of_variation_undefined(self) -> None:
        fairness = client_fairness([0.0, 0.0])

        self.assertIsNone(fairness["coefficient_of_variation"])

    def test_rejects_empty_nonfinite_and_mismatched_inputs(self) -> None:
        with self.assertRaises(ValueError):
            client_fairness([])
        with self.assertRaises(ValueError):
            client_fairness([0.5, float("nan")])
        with self.assertRaises(ValueError):
            client_fairness([0.5, float("inf")])
        with self.assertRaises(ValueError):
            client_fairness([0.5, 0.6], num_examples=[10])
        with self.assertRaises(ValueError):
            client_fairness([0.5, 0.6], num_examples=[10, -1])
        with self.assertRaises(ValueError):
            client_fairness([0.5, 0.6], num_examples=[0, 0])


class GapVsCentralizedTests(unittest.TestCase):
    def test_positive_gap_means_federation_is_behind(self) -> None:
        """The sign convention, pinned. An inverted gap would read as the
        federation beating centralized training — the thesis's central claim
        pointing backwards."""

        self.assertAlmostEqual(gap_vs_centralized(0.80, 0.90), 0.10)

    def test_negative_gap_when_federation_wins(self) -> None:
        self.assertAlmostEqual(gap_vs_centralized(0.92, 0.90), -0.02)

    def test_missing_side_is_none_not_zero(self) -> None:
        """0.0 would claim the federation exactly matched a baseline nobody ran."""

        self.assertIsNone(gap_vs_centralized(None, 0.9))
        self.assertIsNone(gap_vs_centralized(0.9, None))
        self.assertIsNone(gap_vs_centralized(None, None))

    def test_zero_gap_is_reported_when_both_sides_exist(self) -> None:
        self.assertEqual(gap_vs_centralized(0.9, 0.9), 0.0)


if __name__ == "__main__":
    unittest.main()
