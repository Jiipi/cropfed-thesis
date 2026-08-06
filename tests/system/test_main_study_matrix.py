"""The main-study launcher must be able to run everything the thesis claims.

Sections A and B added two skew profiles and three algorithms.  If the launcher
cannot reach them they are unused capability: the results table would still show
only FedAvg and FedProx over label skew, and nobody would notice until the
comparison chapter was already written.

These tests exercise the launcher's pure logic only — scenario selection,
profile resolution and metadata reading.  Actually running Flower belongs to the
GPU study, not to the test suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_main_study.py"
FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


def _load_launcher():
    """Import run_main_study.py, which is a script rather than a package module."""

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_main_study", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class ScenarioMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load_launcher()

    def test_every_implemented_algorithm_is_exercised(self) -> None:
        """B wired five algorithms; a matrix that runs two proves nothing."""

        from typing import get_args

        from cropfed.config import Algorithm

        matrix = {str(s["algorithm"]) for s in self.launcher.SCENARIOS}
        for algorithm in get_args(Algorithm):
            with self.subTest(algorithm=algorithm):
                self.assertIn(
                    algorithm,
                    matrix,
                    f"{algorithm} is implemented but no scenario runs it",
                )

    def test_every_prepared_profile_is_exercised(self) -> None:
        """A prepared six profiles; every one needs at least one scenario."""

        from cropfed.data.profiles import FULL_PROFILE_SPECS

        used_directories = {
            self.launcher.PROFILE_DIRECTORIES[str(s["profile"])]
            for s in self.launcher.SCENARIOS
        }
        for spec in FULL_PROFILE_SPECS:
            with self.subTest(profile=spec.name):
                self.assertIn(
                    spec.name,
                    used_directories,
                    f"profile {spec.name} is prepared but never used",
                )

    def test_scenario_ids_are_unique(self) -> None:
        """Duplicate ids would silently overwrite each other's output dir."""

        ids = [str(s["id"]) for s in self.launcher.SCENARIOS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_scenario_declares_a_resolvable_profile(self) -> None:
        for scenario in self.launcher.SCENARIOS:
            with self.subTest(scenario=scenario["id"]):
                resolved = self.launcher._resolve_profile_dir(
                    str(scenario["profile"]), Path("root")
                )
                self.assertEqual(resolved.parent, Path("root"))

    def test_unknown_profile_names_the_valid_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantity-skew"):
            self.launcher._resolve_profile_dir("does-not-exist", Path("root"))

    def test_federated_scenarios_use_federated_algorithms(self) -> None:
        from cropfed.flower.smoke import SUPPORTED_ALGORITHMS

        for scenario in self.launcher.SCENARIOS:
            if scenario["mode"] != "federated":
                continue
            with self.subTest(scenario=scenario["id"]):
                self.assertIn(str(scenario["algorithm"]), SUPPORTED_ALGORITHMS)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class ProfileMetadataTests(unittest.TestCase):
    """Partition metadata must come from the artifact, not the directory name."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load_launcher()

    def _profile(self, tmp: Path, document: dict) -> Path:
        profile_dir = tmp / document.get("name", "profile")
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "profile.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        return profile_dir

    def test_quantity_skew_is_not_reported_as_iid(self) -> None:
        """The regression this function exists for.

        A quantity-skew profile partitions labels IID, so its spec says
        ``partition_kind='iid'``.  Reporting that verbatim would erase the skew
        from the run manifest, the checkpoint and the protocol lock.
        """

        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            profile_dir = self._profile(
                Path(raw),
                {
                    "name": "quantity-skew",
                    "partition_kind": "iid",
                    "dirichlet_alpha": None,
                    "quantity_skew": True,
                },
            )
            metadata = self.launcher._read_profile_metadata(profile_dir)

        self.assertEqual(metadata["partition_kind"], "quantity_skew")
        self.assertIsNone(metadata["dirichlet_alpha"])

    def test_feature_skew_is_read_from_the_artifact(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            profile_dir = self._profile(
                Path(raw),
                {
                    "name": "feature-skew",
                    "partition_kind": "feature_skew",
                    "dirichlet_alpha": None,
                    "quantity_skew": False,
                    "feature_skew_strength": 0.5,
                },
            )
            metadata = self.launcher._read_profile_metadata(profile_dir)

        self.assertEqual(metadata["partition_kind"], "feature_skew")
        self.assertEqual(metadata["feature_skew_strength"], 0.5)

    def test_dirichlet_alpha_comes_from_the_artifact_not_the_name(self) -> None:
        """A renamed or copied directory must not change the recorded alpha."""

        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            profile_dir = self._profile(
                Path(raw),
                {
                    "name": "renamed-directory",
                    "partition_kind": "dirichlet",
                    "dirichlet_alpha": 0.1,
                    "quantity_skew": False,
                },
            )
            metadata = self.launcher._read_profile_metadata(profile_dir)

        self.assertEqual(metadata["partition_kind"], "dirichlet")
        self.assertEqual(metadata["dirichlet_alpha"], 0.1)

    def test_missing_metadata_says_how_to_regenerate_it(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(FileNotFoundError, "prepare-full-profiles"):
                self.launcher._read_profile_metadata(Path(raw))

    def test_unsupported_partition_kind_is_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            profile_dir = self._profile(
                Path(raw), {"name": "weird", "partition_kind": "hand-made"}
            )
            with self.assertRaisesRegex(ValueError, "unsupported partition_kind"):
                self.launcher._read_profile_metadata(profile_dir)

    def test_real_profiles_on_disk_resolve(self) -> None:
        """Guard against the prepared artifacts drifting from the reader."""

        profiles_root = PROJECT_ROOT / "data" / "flower-profiles-full"
        if not profiles_root.is_dir():
            self.skipTest("prepared profiles are not present in this checkout")
        expected = {
            "iid": "iid",
            "alpha-100": "dirichlet",
            "alpha-0.5": "dirichlet",
            "alpha-0.1": "dirichlet",
            "quantity-skew": "quantity_skew",
            "feature-skew": "feature_skew",
        }
        for profile, partition_kind in expected.items():
            profile_dir = self.launcher._resolve_profile_dir(profile, profiles_root)
            if not (profile_dir / "profile.json").is_file():
                self.skipTest(f"profile {profile} has not been prepared")
            with self.subTest(profile=profile):
                metadata = self.launcher._read_profile_metadata(profile_dir)
                self.assertEqual(metadata["partition_kind"], partition_kind)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower runtime is not installed")
class HyperparameterRecordingTests(unittest.TestCase):
    """The launcher and the server must agree on what gets recorded."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load_launcher()

    def test_only_the_active_algorithms_hyperparameters_are_non_zero(self) -> None:
        cases = {
            "fedavg": {},
            "fedprox": {"proximal_mu": 0.02},
            "fedbn": {},
            "scaffold": {"scaffold_server_lr": 0.9},
            "moon": {"moon_temperature": 0.7, "moon_mu": 2.0},
        }
        for algorithm, expected_non_zero in cases.items():
            with self.subTest(algorithm=algorithm):
                values = self.launcher._algorithm_hyperparameters(
                    algorithm,
                    proximal_mu=0.02,
                    scaffold_server_lr=0.9,
                    moon_temperature=0.7,
                    moon_mu=2.0,
                )
                for name, value in values.items():
                    if name in expected_non_zero:
                        self.assertEqual(value, expected_non_zero[name])
                    else:
                        self.assertEqual(value, 0.0, f"{name} should be inactive")

    def test_launcher_matches_the_server_recording_rule(self) -> None:
        """Divergence here means the launcher verifies a value nobody wrote."""

        from flwr.app import Context, RecordDict

        from cropfed.flower.server_app import _algorithm_hyperparameters

        for algorithm in ("fedavg", "fedprox", "fedbn", "scaffold", "moon"):
            with self.subTest(algorithm=algorithm):
                context = Context(
                    run_id=1,
                    node_id=1,
                    node_config={},
                    state=RecordDict(),
                    run_config={
                        "algorithm": algorithm,
                        "proximal-mu": 0.02,
                        "scaffold-server-lr": 0.9,
                        "moon-temperature": 0.7,
                        "moon-mu": 2.0,
                    },
                )
                self.assertEqual(
                    _algorithm_hyperparameters(context),
                    self.launcher._algorithm_hyperparameters(
                        algorithm,
                        proximal_mu=0.02,
                        scaffold_server_lr=0.9,
                        moon_temperature=0.7,
                        moon_mu=2.0,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
