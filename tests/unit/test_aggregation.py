import unittest

import numpy as np

from cropfed.fl.aggregation import (
    ClientUpdate,
    aggregate,
    fedbn_weighted_average_updates,
    weighted_average_updates,
)


class AggregationTests(unittest.TestCase):
    def test_fedavg_uses_sample_count(self) -> None:
        updates = [
            ClientUpdate(
                client_id="a",
                weights={"w": np.array([1.0, 3.0])},
                num_examples=1,
            ),
            ClientUpdate(
                client_id="b",
                weights={"w": np.array([5.0, 7.0])},
                num_examples=3,
            ),
        ]
        result = weighted_average_updates(updates)
        np.testing.assert_allclose(result["w"], np.array([4.0, 6.0]))

    def test_shape_mismatch_is_rejected(self) -> None:
        updates = [
            ClientUpdate("a", {"w": np.ones(2)}, 1),
            ClientUpdate("b", {"w": np.ones(3)}, 1),
        ]
        with self.assertRaises(ValueError):
            weighted_average_updates(updates)

    def test_non_finite_update_is_rejected(self) -> None:
        updates = [
            ClientUpdate("a", {"w": np.ones(2)}, 1),
            ClientUpdate("b", {"w": np.array([np.nan, 1.0])}, 1),
        ]
        with self.assertRaises(ValueError):
            weighted_average_updates(updates)


class FedBNAggregationTests(unittest.TestCase):
    """FedBN keeps batch-norm statistics local; everything else is averaged."""

    def _bn_updates(self) -> list[ClientUpdate]:
        def weights(scale: float) -> dict:
            return {
                "features.0.weight": np.array([scale, scale]),
                "features.1.weight": np.array([scale]),
                "features.1.bias": np.array([scale]),
                "features.1.running_mean": np.array([scale]),
                "features.1.running_var": np.array([scale]),
                "features.1.num_batches_tracked": np.array([1]),
                "classifier.weight": np.array([scale, scale]),
            }

        return [
            ClientUpdate("a", weights(2.0), 1),
            ClientUpdate("b", weights(6.0), 1),
        ]

    def test_batch_norm_parameters_are_not_averaged(self) -> None:
        result = fedbn_weighted_average_updates(self._bn_updates())

        for bn_key in (
            "features.1.weight",
            "features.1.bias",
            "features.1.running_mean",
            "features.1.running_var",
        ):
            with self.subTest(parameter=bn_key):
                np.testing.assert_allclose(result[bn_key], np.array([2.0]))

    def test_non_batch_norm_parameters_are_averaged(self) -> None:
        result = fedbn_weighted_average_updates(self._bn_updates())

        np.testing.assert_allclose(
            result["features.0.weight"], np.array([4.0, 4.0])
        )
        np.testing.assert_allclose(result["classifier.weight"], np.array([4.0, 4.0]))

    def test_fedbn_differs_from_fedavg_on_batch_norm_layers(self) -> None:
        """The whole point of FedBN: BN layers must not match a FedAvg run."""

        updates = self._bn_updates()
        fedbn = fedbn_weighted_average_updates(updates)
        fedavg = weighted_average_updates(updates)

        self.assertFalse(
            np.allclose(fedbn["features.1.running_mean"], fedavg["features.1.running_mean"]),
            "FedBN averaged the BN statistics, so it is indistinguishable from FedAvg",
        )
        np.testing.assert_allclose(
            fedbn["classifier.weight"], fedavg["classifier.weight"]
        )


class AggregationDispatchTests(unittest.TestCase):
    def test_dispatch_routes_each_algorithm(self) -> None:
        updates = [
            ClientUpdate("a", {"w": np.array([2.0])}, 1),
            ClientUpdate("b", {"w": np.array([6.0])}, 1),
        ]
        for algorithm in ("fedavg", "fedprox", "fedbn", "scaffold", "moon"):
            with self.subTest(algorithm=algorithm):
                result = aggregate(updates, algorithm)
                np.testing.assert_allclose(result["w"], np.array([4.0]))

    def test_unknown_algorithm_is_rejected(self) -> None:
        updates = [ClientUpdate("a", {"w": np.ones(1)}, 1)]
        with self.assertRaisesRegex(ValueError, "unknown algorithm"):
            aggregate(updates, "not-an-algorithm")


if __name__ == "__main__":
    unittest.main()

