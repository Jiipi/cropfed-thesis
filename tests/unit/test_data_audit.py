import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cropfed.constants import TOMATO_CLASSES
from cropfed.data.audit import audit_prepared_data
from cropfed.data.manifest import ImageRecord, write_manifest


class DataAuditTests(unittest.TestCase):
    def test_valid_prepared_data_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_valid_prepared_data(Path(directory))
            report = self._audit(paths)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["global_split_overlap"]["content_sha256_count"], 0)
        self.assertTrue(report["client_assignment"]["complete"])
        self.assertEqual(report["images"]["verified_images"], 20)
        self.assertFalse(report["privacy"]["contains_local_image_paths"])

    def test_cross_split_content_duplicate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_valid_prepared_data(Path(directory))
            paths["test_images"][0].write_bytes(paths["train_images"][0].read_bytes())
            report = self._audit(paths)

        codes = {issue["code"] for issue in report["errors"]}
        self.assertEqual(report["status"], "failed")
        self.assertIn("cross_split_content_overlap", codes)
        self.assertIn("client_global_test_content_overlap", codes)

    def test_corrupt_image_fails_without_leaking_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_valid_prepared_data(Path(directory))
            paths["train_images"][0].write_bytes(b"not an image")
            report = self._audit(paths)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["images"]["invalid_images"])
        self.assertNotIn(str(paths["root"]), str(report))

    def test_duplicate_content_across_local_train_and_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_valid_prepared_data(Path(directory))
            paths["train_images"][8].write_bytes(
                paths["train_images"][0].read_bytes()
            )
            report = self._audit(paths)

        codes = {issue["code"] for issue in report["errors"]}
        self.assertEqual(report["status"], "failed")
        self.assertIn("client_train_validation_content_overlap", codes)
        self.assertIn("duplicate_content_assigned_to_multiple_client_scopes", codes)

    def _audit(self, paths):
        return audit_prepared_data(
            train_manifest=paths["train_manifest"],
            test_manifest=paths["test_manifest"],
            client_data_root=paths["client_root"],
            num_clients=4,
            class_names=TOMATO_CLASSES,
        )

    def _write_valid_prepared_data(self, root: Path):
        image_root = root / "images"
        image_root.mkdir(parents=True)
        train_records: list[ImageRecord] = []
        test_records: list[ImageRecord] = []
        train_images: list[Path] = []
        test_images: list[Path] = []

        for index in range(20):
            image_path = image_root / f"image-{index}.png"
            Image.new(
                "RGB",
                (8, 8),
                (index, (index * 7) % 256, (index * 13) % 256),
            ).save(image_path)
            label_id = index % len(TOMATO_CLASSES)
            split = "train" if index < len(TOMATO_CLASSES) else "test"
            record = ImageRecord(
                image_id=f"image-{index}",
                path=str(image_path.resolve()),
                label_id=label_id,
                label_name=TOMATO_CLASSES[label_id],
                split=split,
            )
            if split == "train":
                train_records.append(record)
                train_images.append(image_path)
            else:
                test_records.append(record)
                test_images.append(image_path)

        processed = root / "processed"
        train_manifest = processed / "train_manifest.csv"
        test_manifest = processed / "test_manifest.csv"
        write_manifest(train_records, train_manifest)
        write_manifest(test_records, test_manifest)

        client_root = root / "clients"
        for client_id in range(4):
            assigned = [
                record
                for index, record in enumerate(train_records)
                if index % 4 == client_id
            ]
            local_train = [self._with_split(record, "local_train") for record in assigned[:-1]]
            local_val = [self._with_split(assigned[-1], "local_val")]
            client_dir = client_root / f"client_{client_id}"
            write_manifest(local_train, client_dir / "train_manifest.csv")
            write_manifest(local_val, client_dir / "val_manifest.csv")

        return {
            "root": root,
            "train_manifest": train_manifest,
            "test_manifest": test_manifest,
            "client_root": client_root,
            "train_images": train_images,
            "test_images": test_images,
        }

    @staticmethod
    def _with_split(record: ImageRecord, split: str) -> ImageRecord:
        return ImageRecord(
            image_id=record.image_id,
            path=record.path,
            label_id=record.label_id,
            label_name=record.label_name,
            split=split,
        )


if __name__ == "__main__":
    unittest.main()
