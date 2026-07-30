import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cropfed.constants import PLANTVILLAGE_FOLDER_TO_CLASS
from cropfed.data.profiles import MVP_PROFILE_SPECS, prepare_mvp_profiles


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

    @staticmethod
    def _write_dataset(dataset: Path) -> None:
        image_number = 0
        for folder_name in PLANTVILLAGE_FOLDER_TO_CLASS:
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
