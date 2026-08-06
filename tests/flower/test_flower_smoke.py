import json
import tempfile
import unittest
from pathlib import Path

from cropfed.flower.smoke import (
    SUPPORTED_ALGORITHMS,
    algorithm_artifact_evidence,
    parse_flower_log_evidence,
    strip_ansi,
    validate_run_artifacts,
)


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

    def test_every_implemented_algorithm_can_be_validated(self) -> None:
        """A run the launcher can start must be a run the validator accepts.

        The regression: the validator only knew fedavg and fedprox, so a
        scaffold run trained for hours and then failed at the last step with
        "algorithm must be 'fedavg' or 'fedprox'".
        """

        from typing import get_args

        from cropfed.config import Algorithm

        for algorithm in get_args(Algorithm):
            with self.subTest(algorithm=algorithm):
                self.assertIn(algorithm, SUPPORTED_ALGORITHMS)

    def test_tracked_strategy_names_are_recognised_for_every_algorithm(self) -> None:
        names = {
            "fedbn": "TrackedFedBN",
            "scaffold": "TrackedSCAFFOLD",
            "moon": "TrackedMOON",
        }
        for algorithm, strategy_name in names.items():
            log = f"""
            Registered 4 nodes
            Starting {strategy_name} strategy:
            aggregate_train: Received 4 results and 0 failures
            aggregate_evaluate: Received 4 results and 0 failures
            """
            with self.subTest(algorithm=algorithm):
                evidence = parse_flower_log_evidence(
                    log,
                    algorithm=algorithm,
                    expected_clients=4,
                    proximal_mu=0.01,
                )
                self.assertTrue(evidence["strategy_started"])

    def test_a_scaffold_run_reporting_a_fedavg_log_is_rejected(self) -> None:
        """The strategy name must match the algorithm the run claims to be."""

        log = """
        Registered 4 nodes
        Starting TrackedFedAvg strategy:
        aggregate_train: Received 4 results and 0 failures
        aggregate_evaluate: Received 4 results and 0 failures
        """

        with self.assertRaisesRegex(RuntimeError, "strategy_started"):
            parse_flower_log_evidence(
                log, algorithm="scaffold", expected_clients=4, proximal_mu=0.0
            )

    def test_unknown_algorithm_lists_the_supported_ones(self) -> None:
        with self.assertRaisesRegex(ValueError, "scaffold"):
            parse_flower_log_evidence(
                "", algorithm="fedwhatever", expected_clients=4, proximal_mu=0.0
            )


class AlgorithmArtifactEvidenceTests(unittest.TestCase):
    """The log proves the strategy started; only metrics prove clients complied.

    FedBN, SCAFFOLD and MOON all aggregate like FedAvg on the server, so a run
    whose clients skipped the algorithm has a flawless log and a results table
    that differs from FedAvg by noise alone.
    """

    def _payload(self, rounds: dict[int, dict]) -> dict:
        return {
            "history": [
                {"round": number, "train": train} for number, train in rounds.items()
            ]
        }

    def test_fedavg_and_fedprox_need_no_extra_state(self) -> None:
        for algorithm in ("fedavg", "fedprox"):
            with self.subTest(algorithm=algorithm):
                result = algorithm_artifact_evidence(
                    self._payload({1: {}}),
                    algorithm=algorithm,
                    expected_clients=4,
                    num_rounds=1,
                )
                self.assertFalse(result["algorithm_state_required"])

    def test_scaffold_requires_every_client_in_every_round(self) -> None:
        payload = self._payload(
            {1: {"scaffold_clients_reporting": 4}, 2: {"scaffold_clients_reporting": 4}}
        )

        result = algorithm_artifact_evidence(
            payload, algorithm="scaffold", expected_clients=4, num_rounds=2
        )

        self.assertEqual(result["rounds_verified"], [1, 2])

    def test_scaffold_with_a_missing_client_is_rejected(self) -> None:
        payload = self._payload(
            {1: {"scaffold_clients_reporting": 4}, 2: {"scaffold_clients_reporting": 3}}
        )

        with self.assertRaisesRegex(RuntimeError, "expected 4"):
            algorithm_artifact_evidence(
                payload, algorithm="scaffold", expected_clients=4, num_rounds=2
            )

    def test_a_scaffold_run_with_no_control_variates_is_rejected(self) -> None:
        """This is the silent FedAvg the whole check exists to catch."""

        payload = self._payload({1: {"train_loss": 1.0}, 2: {"train_loss": 0.9}})

        with self.assertRaisesRegex(RuntimeError, "plain FedAvg"):
            algorithm_artifact_evidence(
                payload, algorithm="scaffold", expected_clients=4, num_rounds=2
            )

    def test_moon_and_fedbn_exempt_only_the_first_round(self) -> None:
        cases = {
            "moon": "moon_contrastive_loss",
            "fedbn": "fedbn_local_bn_tensors",
        }
        for algorithm, key in cases.items():
            with self.subTest(algorithm=algorithm):
                payload = self._payload({1: {}, 2: {key: 0.5}, 3: {key: 0.4}})
                result = algorithm_artifact_evidence(
                    payload, algorithm=algorithm, expected_clients=4, num_rounds=3
                )
                self.assertEqual(result["rounds_verified"], [2, 3])
                self.assertTrue(result["first_round_exempt"])

    def test_a_zero_value_counts_as_no_effect(self) -> None:
        payload = self._payload({1: {}, 2: {"fedbn_local_bn_tensors": 0}})

        with self.assertRaisesRegex(RuntimeError, "had no effect"):
            algorithm_artifact_evidence(
                payload, algorithm="fedbn", expected_clients=4, num_rounds=2
            )

    def test_a_single_round_cannot_demonstrate_moon_or_fedbn(self) -> None:
        for algorithm in ("moon", "fedbn"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(RuntimeError, "single-round"):
                    algorithm_artifact_evidence(
                        self._payload({1: {}}),
                        algorithm=algorithm,
                        expected_clients=4,
                        num_rounds=1,
                    )

    def test_a_truncated_history_is_rejected(self) -> None:
        """A crashed run must not pass by simply having fewer rounds."""

        payload = self._payload({1: {}, 2: {"moon_contrastive_loss": 0.5}})

        with self.assertRaisesRegex(RuntimeError, r"missing rounds \[3\]"):
            algorithm_artifact_evidence(
                payload, algorithm="moon", expected_clients=4, num_rounds=3
            )

    def test_an_absent_history_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "per-round history"):
            algorithm_artifact_evidence(
                {}, algorithm="scaffold", expected_clients=4, num_rounds=2
            )


class HyperparameterVerificationTests(unittest.TestCase):
    """A run that silently fell back to a default is not the configured run.

    ``scaffold-server-lr``, ``moon-temperature`` and ``moon-mu`` all have
    defaults, so a typo in the launcher would produce a complete, valid-looking
    artifact at the wrong settings — and nothing in the results table would say
    so.  The manifest is compared against what was actually requested.
    """

    def _manifest(self, tmp: Path, **overrides: object) -> Path:
        document: dict[str, object] = {
            "algorithm": "scaffold",
            "num_clients": 4,
            "raw_images_received_by_server": False,
            "scaffold_server_lr": 1.0,
        }
        document.update(overrides)
        (tmp / "run_manifest.json").write_text(json.dumps(document), encoding="utf-8")
        return tmp

    def _validate(self, output_dir: Path, **hyperparameters: float) -> None:
        validate_run_artifacts(
            output_dir,
            algorithm="scaffold",
            expected_clients=4,
            proximal_mu=0.0,
            log_text="",
            expected_class_order=(),
            hyperparameters=hyperparameters,
        )

    def test_a_manifest_recorded_at_a_different_setting_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output_dir = self._manifest(Path(raw), scaffold_server_lr=1.0)
            with self.assertRaisesRegex(RuntimeError, "scaffold_server_lr"):
                self._validate(output_dir, scaffold_server_lr=0.5)

    def test_a_hyperparameter_the_run_never_recorded_is_rejected(self) -> None:
        """An older artifact predating the field must not pass by omission."""

        with tempfile.TemporaryDirectory() as raw:
            output_dir = self._manifest(Path(raw))
            with self.assertRaisesRegex(RuntimeError, "moon_mu"):
                self._validate(output_dir, moon_mu=1.0)

    def test_matching_hyperparameters_get_past_the_check(self) -> None:
        """The check must not be the thing that fails a correct run.

        Validation continues to the checkpoint, which this manifest does not
        have — reaching that error proves the hyperparameters were accepted.
        """

        with tempfile.TemporaryDirectory() as raw:
            output_dir = self._manifest(Path(raw), scaffold_server_lr=0.5)
            with self.assertRaises(KeyError):
                self._validate(output_dir, scaffold_server_lr=0.5)


if __name__ == "__main__":
    unittest.main()
