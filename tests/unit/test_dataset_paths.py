"""Manifests must describe the same experiment on a different machine.

The partition artifacts cost hours to build and are prepared on one machine but
consumed on another, so the property under test is not "paths resolve" but
"nothing in a manifest names the machine that wrote it".
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cropfed.data.manifest import ImageRecord, read_manifest, write_manifest
from cropfed.data.paths import (
    DATASET_ROOT_ENVIRONMENT_VARIABLE,
    resolve_dataset_root,
    resolve_image_path,
    to_manifest_path,
)

TORCH_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "torchvision")
)


class DatasetRootResolutionTests(unittest.TestCase):
    def test_explicit_root_beats_the_environment(self) -> None:
        """A stale shell variable must never redirect an explicit --dataset-root."""

        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit"
            explicit.mkdir()
            with mock.patch.dict(
                os.environ,
                {DATASET_ROOT_ENVIRONMENT_VARIABLE: str(Path(directory) / "stale")},
            ):
                self.assertEqual(resolve_dataset_root(explicit), explicit.resolve())

    def test_environment_is_used_when_no_root_is_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(
                os.environ, {DATASET_ROOT_ENVIRONMENT_VARIABLE: str(root)}
            ):
                self.assertEqual(resolve_dataset_root(None), root.resolve())

    def test_blank_values_are_unset_rather_than_the_current_directory(self) -> None:
        with mock.patch.dict(os.environ, {DATASET_ROOT_ENVIRONMENT_VARIABLE: "   "}):
            self.assertIsNone(resolve_dataset_root(None))
        self.assertIsNone(resolve_dataset_root(""))


class ImagePathResolutionTests(unittest.TestCase):
    def test_relative_path_without_a_root_raises_instead_of_guessing(self) -> None:
        """Resolving against the CWD would make a run depend on where it started."""

        with self.assertRaisesRegex(ValueError, "no dataset root"):
            resolve_image_path("Tomato___healthy/image-0.png", None)

    def test_the_same_manifest_resolves_under_two_different_roots(self) -> None:
        """This is the portability guarantee, stated directly."""

        stored = "Tomato___healthy/image-0.png"
        windows_style = Path("F:/project/cropfed-thesis/data/raw")
        container_style = Path("/app/data/raw")

        self.assertEqual(
            resolve_image_path(stored, windows_style),
            windows_style / stored,
        )
        self.assertEqual(
            resolve_image_path(stored, container_style),
            container_style / stored,
        )

    def test_absolute_paths_still_open(self) -> None:
        """Profiles built before the change must not be stranded."""

        absolute = Path(tempfile.gettempdir()).resolve() / "image-0.png"
        self.assertEqual(resolve_image_path(str(absolute), None), absolute)

    def test_manifest_paths_use_posix_separators_on_every_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "Tomato___healthy" / "image-0.png"
            image.parent.mkdir(parents=True)
            image.touch()

            stored = to_manifest_path(image, root)

            self.assertEqual(stored, "Tomato___healthy/image-0.png")
            self.assertNotIn("\\", stored)

    def test_an_image_outside_the_root_is_rejected(self) -> None:
        """Silently storing an absolute path here would produce a stuck profile."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            root.mkdir()
            outside = Path(directory) / "elsewhere" / "image-0.png"
            outside.parent.mkdir()
            outside.touch()

            with self.assertRaisesRegex(ValueError, "not inside the dataset root"):
                to_manifest_path(outside, root)


class ManifestPortabilityTests(unittest.TestCase):
    def test_a_written_manifest_names_no_machine(self) -> None:
        """Round-trip the artifact that actually gets handed to the GPU machine."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            image = root / "Tomato___healthy" / "image-0.png"
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = Path(directory) / "train_manifest.csv"

            write_manifest(
                [
                    ImageRecord(
                        image_id="image-0",
                        path=to_manifest_path(image, root),
                        label_id=0,
                        label_name="Tomato___healthy",
                        split="train",
                    )
                ],
                manifest,
            )

            text = manifest.read_text(encoding="utf-8")
            self.assertNotIn(str(root), text)
            self.assertNotIn(directory, text)

            [record] = read_manifest(manifest)
            self.assertFalse(Path(record.path).is_absolute())
            self.assertTrue(resolve_image_path(record.path, root).is_file())


@unittest.skipUnless(
    TORCH_RUNTIME_AVAILABLE,
    "full Torch/Torchvision runtime is not installed",
)
class DatasetRootReachesTheDataloaderTests(unittest.TestCase):
    def test_missing_root_fails_when_the_dataset_is_built(self) -> None:
        """One clear construction error beats a worker traceback mid-run."""

        from cropfed.data.torch_data import ManifestImageDataset

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._relative_manifest(Path(directory))
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(DATASET_ROOT_ENVIRONMENT_VARIABLE, None)
                with self.assertRaisesRegex(ValueError, "no dataset root"):
                    ManifestImageDataset(manifest, training=False)

    def test_dataset_reads_images_through_the_supplied_root(self) -> None:
        from cropfed.data.torch_data import ManifestImageDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._relative_manifest(root)

            dataset = ManifestImageDataset(
                manifest, training=False, dataset_root=root / "raw"
            )

            self.assertEqual(len(dataset), 1)
            _, label_id = dataset[0]
            self.assertEqual(label_id, 0)

    @staticmethod
    def _relative_manifest(root: Path) -> Path:
        from PIL import Image

        dataset_root = root / "raw"
        image = dataset_root / "Tomato___healthy" / "image-0.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
        manifest = root / "train_manifest.csv"
        write_manifest(
            [
                ImageRecord(
                    image_id="image-0",
                    path=to_manifest_path(image, dataset_root),
                    label_id=0,
                    label_name="Tomato___healthy",
                    split="train",
                )
            ],
            manifest,
        )
        return manifest


if __name__ == "__main__":
    unittest.main()
