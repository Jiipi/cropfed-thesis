import unittest

import numpy as np

from cropfed.data.manifest import ImageRecord, write_client_manifests
from cropfed.data.partitioning import (
    dirichlet_partition,
    feature_skew_partition,
    iid_partition,
    partition_statistics,
)


class PartitioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.repeat(np.arange(10), 40)

    def test_iid_assigns_every_sample_once(self) -> None:
        partitions = iid_partition(self.labels, num_clients=4, seed=7)
        combined = np.concatenate(partitions)
        np.testing.assert_array_equal(np.sort(combined), np.arange(self.labels.size))
        self.assertLessEqual(max(map(len, partitions)) - min(map(len, partitions)), 1)

    def test_dirichlet_is_reproducible_and_complete(self) -> None:
        first = dirichlet_partition(self.labels, 4, alpha=0.1, seed=7)
        second = dirichlet_partition(self.labels, 4, alpha=0.1, seed=7)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        combined = np.concatenate(first)
        np.testing.assert_array_equal(np.sort(combined), np.arange(self.labels.size))

    def test_feature_skew_is_reproducible_and_preserves_all_samples(self) -> None:
        first = feature_skew_partition(self.labels, 4, strength=0.8, seed=7)
        second = feature_skew_partition(self.labels, 4, strength=0.8, seed=7)

        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        combined = np.concatenate(first)
        np.testing.assert_array_equal(np.sort(combined), np.arange(self.labels.size))
        self.assertEqual(len(set(combined)), self.labels.size)

    def test_feature_skew_handles_classes_smaller_than_client_count(self) -> None:
        labels = np.repeat(np.arange(8), 2)
        partitions = feature_skew_partition(labels, 4, strength=0.5, seed=7)

        combined = np.concatenate(partitions)
        np.testing.assert_array_equal(np.sort(combined), np.arange(labels.size))
        self.assertTrue(all(len(partition) > 0 for partition in partitions))

    def test_statistics_preserve_counts(self) -> None:
        partitions = iid_partition(self.labels, 4, seed=7)
        stats = partition_statistics(self.labels, partitions, num_classes=10)
        self.assertEqual(sum(row["num_samples"] for row in stats), self.labels.size)
        self.assertTrue(
            all(sum(row["class_counts"]) == row["num_samples"] for row in stats)
        )

    def test_client_manifests_do_not_overlap(self) -> None:
        import tempfile
        from pathlib import Path

        records = [
            ImageRecord(
                image_id=f"img-{index}",
                path=f"/tmp/img-{index}.jpg",
                label_id=index % 10,
                label_name=str(index % 10),
                split="train",
            )
            for index in range(100)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_client_manifests(
                records,
                root,
                num_clients=4,
                partition_kind="iid",
                seed=3,
                num_classes=10,
            )
            ids: list[str] = []
            for client_id in range(4):
                self.assertTrue(
                    (root / f"client_{client_id}" / "train_manifest.csv").is_file()
                )
                self.assertTrue(
                    (root / f"client_{client_id}" / "val_manifest.csv").is_file()
                )
                for filename in ("train_manifest.csv", "val_manifest.csv"):
                    lines = (root / f"client_{client_id}" / filename).read_text().splitlines()
                    ids.extend(line.split(",", 1)[0] for line in lines[1:])
            self.assertEqual(len(ids), 100)
            self.assertEqual(len(set(ids)), 100)
            self.assertEqual(sum(row["num_samples"] for row in summary), 100)

    def test_quantity_skew_preserves_all_samples(self) -> None:
        from cropfed.data.partitioning import make_partitions

        labels = np.repeat(np.arange(10), 10)  # 100 samples
        partitions = make_partitions(
            labels, num_clients=4, kind="iid", seed=2026, quantity_skew=True
        )
        combined = np.concatenate(partitions)
        self.assertEqual(len(combined), 100)
        self.assertEqual(len(set(combined)), 100)
        np.testing.assert_array_equal(np.sort(combined), np.arange(100))
        sizes = [len(p) for p in partitions]
        self.assertEqual(sum(sizes), 100)
        # Quantity skew should result in unequal sizes
        self.assertNotEqual(max(sizes), min(sizes))

    def test_quantity_skew_refuses_to_destroy_dirichlet_label_skew(self) -> None:
        from cropfed.data.partitioning import make_partitions

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            make_partitions(
                self.labels,
                num_clients=4,
                kind="dirichlet",
                alpha=0.5,
                seed=2026,
                quantity_skew=True,
            )

    def test_skewed_partitions_respect_the_min_size_floor(self) -> None:
        """A client below the floor cannot be split into local train/validation.

        ``write_client_manifests`` partitions content groups and asks for
        ``min_size=2``; a client left with one group raises there instead, which
        is why the floor has to hold inside the partitioners.
        """

        from cropfed.data.partitioning import make_partitions

        # Deliberately tight: 4 clients over 24 samples leaves little slack, so
        # an unclamped skew draw would starve someone.
        labels = np.repeat(np.arange(8), 3)
        for kind in ("quantity_skew", "feature_skew"):
            for seed in range(12):
                partitions = make_partitions(
                    labels,
                    num_clients=4,
                    kind=kind,
                    seed=seed,
                    min_size=2,
                    feature_skew_strength=0.9,
                )
                with self.subTest(kind=kind, seed=seed):
                    self.assertGreaterEqual(min(map(len, partitions)), 2)
                    combined = np.concatenate(partitions)
                    np.testing.assert_array_equal(
                        np.sort(combined), np.arange(labels.size)
                    )

    def test_skew_reports_an_impossible_min_size_instead_of_starving_a_client(
        self,
    ) -> None:
        from cropfed.data.partitioning import make_partitions

        labels = np.repeat(np.arange(3), 2)  # 6 samples
        for kind in ("quantity_skew", "feature_skew"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "min_size"):
                make_partitions(labels, num_clients=4, kind=kind, seed=5, min_size=2)

    def test_duplicate_content_stays_in_one_client_split(self) -> None:
        import tempfile
        from pathlib import Path

        from PIL import Image

        records = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(13):
                image_path = root / f"image-{index}.png"
                Image.new("RGB", (4, 4), (index, index * 3, index * 7)).save(
                    image_path
                )
                label_id = 0 if index < 2 else index % 4
                records.append(
                    ImageRecord(
                        image_id=f"image-{index}",
                        path=str(image_path),
                        label_id=label_id,
                        label_name=str(label_id),
                        split="train",
                    )
                )
            (root / "image-1.png").write_bytes((root / "image-0.png").read_bytes())
            client_root = root / "clients"

            write_client_manifests(
                records,
                client_root,
                num_clients=4,
                partition_kind="iid",
                seed=3,
                num_classes=10,
            )

            scopes = {}
            for client_id in range(4):
                for split_name, filename in (
                    ("train", "train_manifest.csv"),
                    ("validation", "val_manifest.csv"),
                ):
                    lines = (
                        client_root / f"client_{client_id}" / filename
                    ).read_text(encoding="utf-8").splitlines()[1:]
                    for line in lines:
                        scopes[line.split(",", 1)[0]] = (client_id, split_name)

            self.assertEqual(scopes["image-0"], scopes["image-1"])


if __name__ == "__main__":
    unittest.main()
