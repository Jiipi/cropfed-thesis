"""The migration rewrites artifacts that cost hours to rebuild.

The dangerous failure is not a crash but a *successful* rewrite against the
wrong dataset root, which would leave manifests that parse cleanly and point at
the wrong images. The ``image_id`` check is what prevents that, so it gets the
most attention here.
"""

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "migrate_manifest_paths",
    PROJECT_ROOT / "scripts" / "migrate_manifest_paths.py",
)
migrate_manifest_paths = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate_manifest_paths)


class ManifestMigrationTests(unittest.TestCase):
    def test_absolute_paths_become_relative_and_ids_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = self._build_profile_set(Path(directory))

            report = migrate_manifest_paths.migrate_profile_set(
                layout["profiles_root"],
                dataset_root=layout["dataset_root"],
                dry_run=False,
            )

            self.assertEqual(report["total_rewritten"], 8)
            rows = self._read(layout["profiles_root"] / "iid" / "train_manifest.csv")
            for row in rows:
                self.assertFalse(Path(row["path"]).is_absolute())
                self.assertEqual(
                    row["image_id"],
                    hashlib.sha1(row["path"].encode("utf-8")).hexdigest()[:16],
                )
                self.assertTrue((layout["dataset_root"] / row["path"]).is_file())

    def test_a_wrong_dataset_root_aborts_before_writing_anything(self) -> None:
        """A plausible-but-wrong root is the failure this check exists to catch."""

        with tempfile.TemporaryDirectory() as directory:
            layout = self._build_profile_set(Path(directory))
            manifest = layout["profiles_root"] / "iid" / "train_manifest.csv"
            before = manifest.read_bytes()
            wrong_root = layout["dataset_root"].parent

            with self.assertRaises(SystemExit) as raised:
                migrate_manifest_paths.migrate_profile_set(
                    layout["profiles_root"], dataset_root=wrong_root, dry_run=False
                )

            self.assertIn("image_id check", str(raised.exception))
            self.assertEqual(manifest.read_bytes(), before)

    def test_dry_run_leaves_every_manifest_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = self._build_profile_set(Path(directory))
            before = {
                path: path.read_bytes()
                for path in layout["profiles_root"].rglob("*.csv")
            }

            report = migrate_manifest_paths.migrate_profile_set(
                layout["profiles_root"],
                dataset_root=layout["dataset_root"],
                dry_run=True,
            )

            self.assertEqual(report["total_rewritten"], 8)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_recorded_checksums_are_updated_to_match_the_new_bytes(self) -> None:
        """A stale checksum would make extend-full-profiles refuse to run."""

        with tempfile.TemporaryDirectory() as directory:
            layout = self._build_profile_set(Path(directory))

            migrate_manifest_paths.migrate_profile_set(
                layout["profiles_root"],
                dataset_root=layout["dataset_root"],
                dry_run=False,
            )

            index = json.loads(
                (layout["profiles_root"] / "profiles_index.json").read_text(
                    encoding="utf-8"
                )
            )
            row = index["profiles"][0]
            actual = migrate_manifest_paths._sha256_file(
                layout["profiles_root"] / "iid" / "train_manifest.csv"
            )
            self.assertEqual(row["train_manifest_sha256"], actual)
            self.assertEqual(index["manifest_paths"], "relative_to_dataset_root")
            self.assertTrue(index["shared_split_invariants"]["same_train_manifest"])

    def test_verify_passes_after_migration_and_fails_before_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = self._build_profile_set(Path(directory))

            before = migrate_manifest_paths.verify_profile_set(
                layout["profiles_root"], dataset_root=layout["dataset_root"]
            )
            self.assertEqual(before["status"], "failed")

            migrate_manifest_paths.migrate_profile_set(
                layout["profiles_root"],
                dataset_root=layout["dataset_root"],
                dry_run=False,
            )
            after = migrate_manifest_paths.verify_profile_set(
                layout["profiles_root"], dataset_root=layout["dataset_root"]
            )

            self.assertEqual(after["status"], "passed")
            self.assertEqual(after["missing_files"], 0)
            self.assertEqual(after["findings"], [])

    def test_migration_is_idempotent(self) -> None:
        """Re-running after a partial interruption must not corrupt anything."""

        with tempfile.TemporaryDirectory() as directory:
            layout = self._build_profile_set(Path(directory))
            migrate_manifest_paths.migrate_profile_set(
                layout["profiles_root"],
                dataset_root=layout["dataset_root"],
                dry_run=False,
            )
            once = (
                layout["profiles_root"] / "iid" / "train_manifest.csv"
            ).read_bytes()

            second = migrate_manifest_paths.migrate_profile_set(
                layout["profiles_root"],
                dataset_root=layout["dataset_root"],
                dry_run=False,
            )

            self.assertEqual(second["total_rewritten"], 0)
            self.assertEqual(
                (layout["profiles_root"] / "iid" / "train_manifest.csv").read_bytes(),
                once,
            )

    def test_client_manifests_are_migrated_too(self) -> None:
        """Clients hold the rows that training actually reads."""

        with tempfile.TemporaryDirectory() as directory:
            layout = self._build_profile_set(Path(directory))

            migrate_manifest_paths.migrate_profile_set(
                layout["profiles_root"],
                dataset_root=layout["dataset_root"],
                dry_run=False,
            )

            client = (
                layout["profiles_root"]
                / "iid"
                / "clients"
                / "client_0"
                / "train_manifest.csv"
            )
            for row in self._read(client):
                self.assertFalse(Path(row["path"]).is_absolute())
                self.assertTrue((layout["dataset_root"] / row["path"]).is_file())

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _build_profile_set(self, root: Path) -> dict[str, Path]:
        """Build a miniature profile set whose ids match the real convention."""

        dataset_root = root / "raw" / "color"
        profiles_root = root / "profiles"
        profile_root = profiles_root / "iid"
        client_root = profile_root / "clients" / "client_0"
        client_root.mkdir(parents=True)

        records = []
        for index in range(2):
            relative = f"Apple___Apple_scab/image-{index}.JPG"
            image = dataset_root / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fake image bytes")
            records.append(
                {
                    "image_id": hashlib.sha1(
                        relative.encode("utf-8")
                    ).hexdigest()[:16],
                    "path": str(image.resolve()),
                    "label_id": "0",
                    "label_name": "Apple - Apple scab",
                    "split": "train",
                }
            )

        for target, split in (
            (profile_root / "train_manifest.csv", "train"),
            (profile_root / "test_manifest.csv", "test"),
            (client_root / "train_manifest.csv", "local_train"),
            (client_root / "val_manifest.csv", "local_val"),
        ):
            self._write(target, records, split)

        index = {
            "schema_version": 1,
            "num_clients": 4,
            "seed": 2026,
            "shared_split_invariants": {},
            "profiles": [
                {
                    "name": "iid",
                    "train_manifest_sha256": migrate_manifest_paths._sha256_file(
                        profile_root / "train_manifest.csv"
                    ),
                    "test_manifest_sha256": migrate_manifest_paths._sha256_file(
                        profile_root / "test_manifest.csv"
                    ),
                }
            ],
        }
        (profiles_root / "profiles_index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
        (profile_root / "profile.json").write_text(
            json.dumps({"name": "iid"}, indent=2), encoding="utf-8"
        )
        return {
            "profiles_root": profiles_root,
            "dataset_root": dataset_root.resolve(),
        }

    @staticmethod
    def _write(path: Path, records: list[dict[str, str]], split: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "path", "label_id", "label_name", "split"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow({**record, "split": split})


if __name__ == "__main__":
    unittest.main()
