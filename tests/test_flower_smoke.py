import unittest

from cropfed.flower.smoke import parse_flower_log_evidence, strip_ansi


class FlowerSmokeEvidenceTests(unittest.TestCase):
    def test_fedavg_four_client_evidence(self) -> None:
        log = """
        Registered 4 nodes
        Starting FedAvg strategy:
        aggregate_train: Received 4 results and 0 failures
        aggregate_evaluate: Received 4 results and 0 failures
        """

        evidence = parse_flower_log_evidence(
            log,
            algorithm="fedavg",
            expected_clients=4,
            proximal_mu=0.01,
        )

        self.assertTrue(all(evidence.values()))

    def test_fedprox_requires_mu_evidence(self) -> None:
        log = """
        (4 simulated SuperNodes)
        Starting FedProx strategy:
        Proximal mu: 0.01
        aggregate_train: Received 4 results and 0 failures
        aggregate_evaluate: Received 4 results and 0 failures
        """

        evidence = parse_flower_log_evidence(
            log,
            algorithm="fedprox",
            expected_clients=4,
            proximal_mu=0.01,
        )

        self.assertTrue(evidence["proximal_mu_confirmed"])

    def test_tracked_strategy_name_is_accepted(self) -> None:
        log = """
        Registered 4 nodes
        Starting TrackedFedAvg strategy:
        aggregate_train: Received 4 results and 0 failures
        aggregate_evaluate: Received 4 results and 0 failures
        """

        evidence = parse_flower_log_evidence(
            log,
            algorithm="fedavg",
            expected_clients=4,
            proximal_mu=0.01,
        )

        self.assertTrue(evidence["strategy_started"])

    def test_missing_client_result_is_rejected(self) -> None:
        log = """
        Registered 4 nodes
        Starting FedAvg strategy:
        aggregate_train: Received 3 results and 1 failures
        aggregate_evaluate: Received 4 results and 0 failures
        """

        with self.assertRaisesRegex(RuntimeError, "train_results_complete"):
            parse_flower_log_evidence(
                log,
                algorithm="fedavg",
                expected_clients=4,
                proximal_mu=0.01,
            )

    def test_strip_ansi(self) -> None:
        self.assertEqual(strip_ansi("\x1b[92mINFO\x1b[0m"), "INFO")


if __name__ == "__main__":
    unittest.main()
