"""A generated lock is worthless unless the runner's validator accepts it.

The failure this file guards against is silent and expensive: a lock whose
config differs from what the experiment presents at validation time is rejected
only after the run reaches its validation step, so the GPU time is already
spent. The config dicts are therefore compared against the shapes the consumers
build, not merely round-tripped through the generator.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "generate_protocol_locks",
    PROJECT_ROOT / "scripts" / "generate_protocol_locks.py",
)
generate_protocol_locks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generate_protocol_locks)

from cropfed.experiments.protocol import validate_protocol_lock  # noqa: E402

DEFAULTS = {
    "rounds": 10,
    "local_epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.001,
    "model": "mobilenet_v3_small",
    "pretrained": True,
    "proximal_mu": 0.01,
    "scaffold_server_lr": 1.0,
    "moon_temperature": 0.5,
    "moon_mu": 1.0,
}


class ProtocolLockGeneratorTests(unittest.TestCase):
    def test_one_lock_per_scenario_named_for_the_runner(self) -> None:
        """run_main_study looks up ``<scenario-id>.json`` in lower case."""

        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"

            report = generate_protocol_locks.generate_locks(
                output,
                profiles_root=profiles,
                allowed_seeds=[2026],
                **DEFAULTS,
            )

            study = generate_protocol_locks._load_main_study()
            self.assertEqual(report["num_locks"], len(study.SCENARIOS))
            for scenario in study.SCENARIOS:
                expected = output / f"{str(scenario['id']).lower()}.json"
                self.assertTrue(expected.is_file(), f"missing lock {expected.name}")

    def test_every_generated_lock_passes_its_own_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026], **DEFAULTS
            )

            for lock_path in sorted(output.glob("*.json")):
                document = json.loads(lock_path.read_text(encoding="utf-8"))
                with self.subTest(lock=lock_path.name):
                    result = validate_protocol_lock(
                        lock_path,
                        experiment_type=document["experiment_type"],
                        config=document["config"],
                        manifest_hashes=document["manifest_hashes"],
                        seed=2026,
                    )
                    self.assertEqual(result["status"], "locked")

    def test_federated_config_matches_the_server_app_contract(self) -> None:
        """The four hyperparameters must be present and zero when inapplicable."""

        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026], **DEFAULTS
            )

            scaffold = json.loads(
                (output / "fl-a01-scaf.json").read_text(encoding="utf-8")
            )["config"]

            self.assertEqual(
                scaffold,
                {
                    "algorithm": "scaffold",
                    "partition_kind": "dirichlet",
                    "dirichlet_alpha": 0.1,
                    "model": "mobilenet_v3_small",
                    "num_clients": 4,
                    "num_rounds": 10,
                    "local_epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "pretrained": True,
                    "proximal_mu": 0.0,
                    "scaffold_server_lr": 1.0,
                    "moon_temperature": 0.0,
                    "moon_mu": 0.0,
                },
            )

    def test_an_iid_profile_locks_alpha_as_zero_not_null(self) -> None:
        """The runner writes ``dirichlet_alpha or 0.0``, then the server casts it."""

        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026], **DEFAULTS
            )

            config = json.loads(
                (output / "fl-iid-avg.json").read_text(encoding="utf-8")
            )["config"]

            self.assertEqual(config["dirichlet_alpha"], 0.0)
            self.assertIsNotNone(config["dirichlet_alpha"])

    def test_quantity_skew_is_locked_as_quantity_skew(self) -> None:
        """A quantity-skew profile partitions IID; the lock must not say 'iid'."""

        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026], **DEFAULTS
            )

            config = json.loads(
                (output / "fl-qty-avg.json").read_text(encoding="utf-8")
            )["config"]

            self.assertEqual(config["partition_kind"], "quantity_skew")

    def test_manifest_hashes_are_the_real_file_digests(self) -> None:
        from cropfed.experiments.artifacts import file_sha256

        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026], **DEFAULTS
            )

            lock = json.loads(
                (output / "fl-a01-avg.json").read_text(encoding="utf-8")
            )
            profile = profiles / "dirichlet-alpha-0.1"

            self.assertEqual(
                lock["manifest_hashes"]["global_test_sha256"],
                file_sha256(profile / "test_manifest.csv"),
            )
            self.assertEqual(
                lock["manifest_hashes"]["partition_summary_sha256"],
                file_sha256(profile / "clients" / "partition_summary.json"),
            )

    def test_centralized_lock_records_the_pooled_train_manifest(self) -> None:
        """Centralized trains on pooled data, so locking the master split would
        validate against a file the run never reads."""

        from cropfed.experiments.artifacts import file_sha256

        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026], **DEFAULTS
            )

            lock = json.loads((output / "cen-mbv3.json").read_text(encoding="utf-8"))

            self.assertEqual(lock["experiment_type"], "centralized")
            self.assertEqual(
                lock["manifest_hashes"]["train_sha256"],
                file_sha256(profiles / "iid" / "pooled_train_manifest.csv"),
            )
            self.assertEqual(
                sorted(lock["manifest_hashes"]),
                ["test_sha256", "train_sha256", "validation_sha256"],
            )

    def test_a_changed_manifest_invalidates_the_lock(self) -> None:
        """This is the property the whole mechanism exists to provide."""

        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026], **DEFAULTS
            )
            lock_path = output / "fl-a01-avg.json"
            document = json.loads(lock_path.read_text(encoding="utf-8"))

            tampered = dict(document["manifest_hashes"])
            tampered["global_test_sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "differ from protocol lock"):
                validate_protocol_lock(
                    lock_path,
                    experiment_type=document["experiment_type"],
                    config=document["config"],
                    manifest_hashes=tampered,
                    seed=2026,
                )

    def test_a_seed_outside_allowed_seeds_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiles = self._build_profiles(Path(directory) / "profiles")
            output = Path(directory) / "locks"
            generate_protocol_locks.generate_locks(
                output, profiles_root=profiles, allowed_seeds=[2026, 2027], **DEFAULTS
            )
            lock_path = output / "fl-a01-avg.json"
            document = json.loads(lock_path.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValueError, "allowed_seeds"):
                validate_protocol_lock(
                    lock_path,
                    experiment_type=document["experiment_type"],
                    config=document["config"],
                    manifest_hashes=document["manifest_hashes"],
                    seed=9999,
                )

    def test_a_missing_profile_fails_before_writing_locks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                generate_protocol_locks.generate_locks(
                    Path(directory) / "locks",
                    profiles_root=Path(directory) / "absent",
                    allowed_seeds=[2026],
                    **DEFAULTS,
                )

    def _build_profiles(self, root: Path) -> Path:
        """Create the six profile directories the scenario matrix references."""

        names = {
            "iid": ("iid", None, False),
            "dirichlet-alpha-100": ("dirichlet", 100.0, False),
            "dirichlet-alpha-0.5": ("dirichlet", 0.5, False),
            "dirichlet-alpha-0.1": ("dirichlet", 0.1, False),
            "quantity-skew": ("iid", None, True),
            "feature-skew": ("feature_skew", None, False),
        }
        for name, (kind, alpha, quantity_skew) in names.items():
            profile = root / name
            (profile / "clients").mkdir(parents=True)
            (profile / "profile.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "partition_kind": kind,
                        "dirichlet_alpha": alpha,
                        "quantity_skew": quantity_skew,
                        "feature_skew_strength": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            for manifest in (
                "train_manifest.csv",
                "test_manifest.csv",
                "pooled_train_manifest.csv",
                "validation_manifest.csv",
            ):
                (profile / manifest).write_text(
                    f"image_id,path,label_id,label_name,split\nx,{name}.JPG,0,a,train\n",
                    encoding="utf-8",
                )
            (profile / "clients" / "partition_summary.json").write_text(
                json.dumps({"clients": [], "skew_type": kind}), encoding="utf-8"
            )
        return root


if __name__ == "__main__":
    unittest.main()
