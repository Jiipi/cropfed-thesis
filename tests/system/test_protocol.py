import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cropfed.experiments.protocol import validate_protocol_lock


class ProtocolLockTests(unittest.TestCase):
    def test_accepts_exact_locked_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "protocol-lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "locked",
                        "experiment_type": "centralized",
                        "config": {"model": "mobilenet_v2", "epochs": 30},
                        "manifest_hashes": {"train_sha256": "abc"},
                        "allowed_seeds": [2026, 2027, 2028],
                    }
                ),
                encoding="utf-8",
            )

            result = validate_protocol_lock(
                lock_path,
                experiment_type="centralized",
                config={"model": "mobilenet_v2", "epochs": 30},
                manifest_hashes={"train_sha256": "abc"},
                seed=2027,
            )

            expected_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            self.assertEqual(result["status"], "locked")
            self.assertEqual(result["sha256"], expected_hash)

    def test_rejects_draft_or_changed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "protocol-lock.json"
            payload = {
                "schema_version": 1,
                "status": "draft",
                "experiment_type": "federated",
                "config": {"num_rounds": 30},
                "manifest_hashes": {"global_test_sha256": "abc"},
                "allowed_seeds": [2026],
            }
            lock_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "status"):
                validate_protocol_lock(
                    lock_path,
                    experiment_type="federated",
                    config={"num_rounds": 30},
                    manifest_hashes={"global_test_sha256": "abc"},
                    seed=2026,
                )

            payload["status"] = "locked"
            lock_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run config differs"):
                validate_protocol_lock(
                    lock_path,
                    experiment_type="federated",
                    config={"num_rounds": 20},
                    manifest_hashes={"global_test_sha256": "abc"},
                    seed=2026,
                )


if __name__ == "__main__":
    unittest.main()
