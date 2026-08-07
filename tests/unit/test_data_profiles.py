import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cropfed.constants import (
    PLANTVILLAGE_FOLDER_TO_CLASS,
    PLANTVILLAGE_FULL_FOLDER_TO_CLASS,
    PLANTVILLAGE_FULL_TAXONOMY,
    TOMATO_TAXONOMY,
)
from cropfed.data.manifest import read_manifest
from cropfed.data.profiles import (
    FULL_PROFILE_SPECS,
    MVP_PROFILE_SPECS,
    DataProfileSpec,
    extend_data_profiles,
    prepare_full_profiles,
    prepare_mvp_profiles,
)


class DataProfileTests(unittest.TestCase):
    def test_three_profiles_share_global_split_and_pass_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "plantvillage"
            self._write_dataset(dataset)

            result = prepare_mvp_profiles(
                dataset_root=dataset,
                output_root=root / "profiles",
                num_clients=4,
                seed=11,
            )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["profiles"]), 3)
            self.assertTrue(
                result["shared_split_invariants"]["same_global_test_manifest"]
            )
            self.assertTrue(result["shared_split_invariants"]["same_train_manifest"])
            self.assertEqual(
                [profile["name"] for profile in result["profiles"]],
                [spec.name for spec in MVP_PROFILE_SPECS],
            )
            self.assertTrue(
                all(profile["audit_status"] == "passed" for profile in result["profiles"])
            )
            self.assertEqual(
                result["content_grouped_split"]["num_duplicate_content_groups"],
                1,
            )
            for spec in MVP_PROFILE_SPECS:
                profile_root = root / "profiles" / spec.name
                master_ids = {
                    row.image_id
                    for row in read_manifest(profile_root / "train_manifest.csv")
                }
                pooled_train_ids = {
                    row.image_id
                    for row in read_manifest(
                        profile_root / "pooled_train_manifest.csv"
                    )
                }
                validation_ids = {
                    row.image_id
                    for row in read_manifest(
                        profile_root / "validation_manifest.csv"
                    )
                }
                self.assertFalse(pooled_train_ids & validation_ids)
                self.assertEqual(pooled_train_ids | validation_ids, master_ids)

    def test_full_taxonomy_profiles_cover_all_38_classes(self) -> None:
        """The 38-class main-study path must produce its own four profiles.

        ``prepare_full_profiles`` had no coverage while the tomato pilot was the
        only scope, so a regression there would only surface during a multi-hour
        GPU run.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "plantvillage-full"
            self._write_dataset(dataset, PLANTVILLAGE_FULL_FOLDER_TO_CLASS)

            result = prepare_full_profiles(
                dataset_root=dataset,
                output_root=root / "profiles",
                num_clients=4,
                seed=11,
            )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["taxonomy_scope"], "plantvillage-full")
            self.assertEqual(result["num_classes"], 38)
            self.assertEqual(
                result["class_order"], list(PLANTVILLAGE_FULL_TAXONOMY.class_names)
            )
            self.assertEqual(
                [profile["name"] for profile in result["profiles"]],
                [spec.name for spec in FULL_PROFILE_SPECS],
            )
            self.assertTrue(
                result["shared_split_invariants"]["same_global_test_manifest"]
            )
            self.assertTrue(
                all(
                    profile["audit_status"] == "passed"
                    for profile in result["profiles"]
                )
            )
            # Labels must span the whole taxonomy, not just its first ten IDs.
            label_ids = {
                row.label_id
                for row in read_manifest(
                    root / "profiles" / "iid" / "train_manifest.csv"
                )
            }
            self.assertEqual(label_ids, set(range(38)))

    def test_refuses_to_overwrite_non_empty_profile_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_mvp_profiles(
                    dataset_root=root / "not-needed",
                    output_root=output,
                )

            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "keep")

    def test_full_profile_set_covers_all_three_proposal_skews(self) -> None:
        """§5 GĐ2 asks for label, quantity and feature skew — not label alone."""

        kinds = {
            (spec.partition_kind, spec.quantity_skew) for spec in FULL_PROFILE_SPECS
        }
        self.assertIn(("dirichlet", False), kinds)  # label skew
        self.assertIn(("iid", True), kinds)  # quantity skew
        self.assertIn(("feature_skew", False), kinds)  # feature skew

    def test_quantity_and_feature_profiles_are_partitioned_as_declared(self) -> None:
        """The declared skew must reach the partitioner, not just the spec.

        Two of the three skews used to be dead code: ``make_partitions`` accepted
        them but nothing in the profile pipeline ever passed them, so a profile
        named ``quantity-skew`` was partitioned as plain IID.
        """

        specs = (
            DataProfileSpec("iid", "iid", None),
            DataProfileSpec("quantity-skew", "iid", None, quantity_skew=True),
            DataProfileSpec("feature-skew", "feature_skew", None),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "plantvillage"
            self._write_dataset(dataset)

            from cropfed.data.profiles import prepare_data_profiles

            result = prepare_data_profiles(
                dataset_root=dataset,
                output_root=root / "profiles",
                taxonomy=TOMATO_TAXONOMY,
                profile_specs=specs,
                num_clients=4,
                seed=11,
            )

            self.assertEqual(result["status"], "passed")
            summaries = {
                spec.name: json.loads(
                    (
                        root / "profiles" / spec.name / "clients" / "partition_summary.json"
                    ).read_text(encoding="utf-8")
                )
                for spec in specs
            }

            # A3: the artifact must describe which skew produced it.
            self.assertEqual(summaries["iid"]["skew_type"], "none")
            self.assertEqual(summaries["quantity-skew"]["skew_type"], "quantity")
            self.assertEqual(summaries["feature-skew"]["skew_type"], "feature")
            self.assertTrue(summaries["quantity-skew"]["quantity_skew"])
            self.assertEqual(summaries["feature-skew"]["feature_skew_strength"], 0.5)

            def sizes(name: str) -> list[int]:
                return [row["num_samples"] for row in summaries[name]["clients"]]

            # Quantity skew must produce genuinely unequal client sizes, which is
            # exactly what an accidental IID fallback would not do.
            self.assertGreater(max(sizes("quantity-skew")), min(sizes("quantity-skew")))
            self.assertNotEqual(sorted(sizes("quantity-skew")), sorted(sizes("iid")))
            self.assertEqual(sum(sizes("quantity-skew")), sum(sizes("iid")))
            self.assertEqual(sum(sizes("feature-skew")), sum(sizes("iid")))

    def test_extension_reuses_the_existing_split_and_keeps_d024(self) -> None:
        """D-024: adding a profile must not create a second global test set."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "plantvillage"
            output = root / "profiles"
            self._write_dataset(dataset)

            original = prepare_mvp_profiles(
                dataset_root=dataset, output_root=output, num_clients=4, seed=11
            )
            original_test_sha = original["shared_split_invariants"][
                "test_manifest_sha256"
            ]
            original_names = [row["name"] for row in original["profiles"]]

            extended = extend_data_profiles(
                output_root=output,
                taxonomy=TOMATO_TAXONOMY,
                profile_specs=(
                    DataProfileSpec("quantity-skew", "iid", None, quantity_skew=True),
                    DataProfileSpec("feature-skew", "feature_skew", None),
                ),
                num_clients=4,
                seed=11,
                dataset_root=dataset,
            )

            self.assertEqual(extended["status"], "passed")
            self.assertEqual(
                [row["name"] for row in extended["profiles"]],
                [*original_names, "quantity-skew", "feature-skew"],
            )
            invariants = extended["shared_split_invariants"]
            self.assertTrue(invariants["same_global_test_manifest"])
            self.assertTrue(invariants["same_train_manifest"])
            self.assertEqual(invariants["test_manifest_sha256"], original_test_sha)

            # The point of the constraint: byte-identical manifests, not merely
            # manifests that happen to hold the same number of rows.
            for name in ("quantity-skew", "feature-skew"):
                self.assertEqual(
                    (output / name / "test_manifest.csv").read_bytes(),
                    (output / "iid" / "test_manifest.csv").read_bytes(),
                )
                self.assertEqual(
                    (output / name / "train_manifest.csv").read_bytes(),
                    (output / "iid" / "train_manifest.csv").read_bytes(),
                )

            index = json.loads(
                (output / "profiles_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(index["profiles"]), len(original_names) + 2)

    def test_extension_refuses_to_overwrite_an_existing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "plantvillage"
            output = root / "profiles"
            self._write_dataset(dataset)
            prepare_mvp_profiles(
                dataset_root=dataset, output_root=output, num_clients=4, seed=11
            )
            before = (output / "iid" / "clients" / "partition_summary.json").read_bytes()

            with self.assertRaises(FileExistsError):
                extend_data_profiles(
                    output_root=output,
                    taxonomy=TOMATO_TAXONOMY,
                    profile_specs=(DataProfileSpec("iid", "feature_skew", None),),
                    num_clients=4,
                    seed=11,
                    dataset_root=dataset,
                )

            self.assertEqual(
                (output / "iid" / "clients" / "partition_summary.json").read_bytes(),
                before,
            )

    def test_extension_rejects_a_different_seed(self) -> None:
        """A different seed means a different partition basis, so comparisons break."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "plantvillage"
            output = root / "profiles"
            self._write_dataset(dataset)
            prepare_mvp_profiles(
                dataset_root=dataset, output_root=output, num_clients=4, seed=11
            )

            with self.assertRaisesRegex(ValueError, "seed"):
                extend_data_profiles(
                    output_root=output,
                    taxonomy=TOMATO_TAXONOMY,
                    profile_specs=(DataProfileSpec("feature-skew", "feature_skew", None),),
                    num_clients=4,
                    seed=99,
                    dataset_root=dataset,
                )

    def test_extension_detects_a_tampered_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "plantvillage"
            output = root / "profiles"
            self._write_dataset(dataset)
            prepare_mvp_profiles(
                dataset_root=dataset, output_root=output, num_clients=4, seed=11
            )
            manifest = output / "iid" / "test_manifest.csv"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").rsplit("\n", 2)[0] + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "checksum"):
                extend_data_profiles(
                    output_root=output,
                    taxonomy=TOMATO_TAXONOMY,
                    profile_specs=(DataProfileSpec("feature-skew", "feature_skew", None),),
                    num_clients=4,
                    seed=11,
                    dataset_root=dataset,
                )

    @staticmethod
    def _write_dataset(dataset: Path, folders=PLANTVILLAGE_FOLDER_TO_CLASS) -> None:
        image_number = 0
        for folder_name in folders:
            class_dir = dataset / folder_name
            class_dir.mkdir(parents=True)
            for local_index in range(4):
                image_path = class_dir / f"image-{local_index}.png"
                Image.new(
                    "RGB",
                    (8, 8),
                    (
                        image_number % 256,
                        (image_number * 17) % 256,
                        (image_number * 29) % 256,
                    ),
                ).save(image_path)
                image_number += 1
        healthy = dataset / "Tomato___healthy"
        (healthy / "image-3.png").write_bytes((healthy / "image-0.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
