import unittest

import numpy as np

from cropfed.data.manifest import ImageRecord, write_client_manifests
from cropfed.data.partitioning import (
    dirichlet_partition,
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
            )
            ids: list[str] = []
            for client_id in range(4):
                for filename in ("train_manifest.csv", "val_manifest.csv"):
                    lines = (root / f"client_{client_id}" / filename).read_text().splitlines()
                    ids.extend(line.split(",", 1)[0] for line in lines[1:])
            self.assertEqual(len(ids), 100)
            self.assertEqual(len(set(ids)), 100)
            self.assertEqual(sum(row["num_samples"] for row in summary), 100)


if __name__ == "__main__":
    unittest.main()
