import unittest

import numpy as np

from cropfed.fl.aggregation import ClientUpdate, weighted_average_updates


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


if __name__ == "__main__":
    unittest.main()

