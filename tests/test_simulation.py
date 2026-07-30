import unittest

from cropfed.config import ExperimentConfig
from cropfed.simulation import run_synthetic_experiment


class SimulationTests(unittest.TestCase):
    def test_fedavg_smoke_completes(self) -> None:
        config = ExperimentConfig(
            algorithm="fedavg",
            partition_kind="dirichlet",
            num_clients=4,
            num_rounds=2,
            local_epochs=1,
            dirichlet_alpha=0.5,
            seed=11,
        )
        result = run_synthetic_experiment(
            config, samples_per_class=20, num_features=8
        )
        self.assertEqual(result["result_kind"], "synthetic_smoke_only")
        self.assertEqual(len(result["history"]), 2)
        self.assertGreater(result["communication"]["total_bytes"], 0)
        self.assertIn("macro_f1", result["final_metrics"])

    def test_fedprox_smoke_completes(self) -> None:
        config = ExperimentConfig(
            algorithm="fedprox",
            partition_kind="iid",
            num_clients=4,
            num_rounds=1,
            local_epochs=1,
            proximal_mu=0.1,
            seed=12,
        )
        result = run_synthetic_experiment(
            config, samples_per_class=20, num_features=8
        )
        self.assertEqual(result["config"]["algorithm"], "fedprox")


if __name__ == "__main__":
    unittest.main()

