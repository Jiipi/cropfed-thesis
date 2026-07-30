import csv
import json
import tempfile
import unittest
from pathlib import Path

from cropfed.constants import TOMATO_CLASSES
from cropfed.experiments.export import export_results
from cropfed.ml.metrics import classification_metrics


class ResearchExportTests(unittest.TestCase):
    def test_export_includes_baseline_and_excludes_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centralized = root / "centralized"
            centralized.mkdir()
            metrics = classification_metrics(
                list(range(10)),
                list(range(10)),
                num_classes=10,
                class_names=TOMATO_CLASSES,
                healthy_class_id=0,
                class_groups=["healthy", *("disease" for _ in range(8)), "pest"],
            )
            (centralized / "result.json").write_text(
                json.dumps(
                    {
                        "experiment_type": "centralized",
                        "model": "mobilenet_v2",
                        "seed": 2026,
                        "metrics": metrics,
                        "elapsed_seconds": 1.5,
                        "checkpoint_bytes": 123,
                    }
                ),
                encoding="utf-8",
            )
            synthetic = root / "synthetic.json"
            synthetic.write_text(
                json.dumps(
                    {
                        "result_kind": "synthetic_smoke_only",
                        "algorithm": "fedavg",
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "export"

            result = export_results(
                [centralized, synthetic],
                output,
                project_root=root,
            )

            self.assertEqual(len(result["included"]), 1)
            self.assertEqual(len(result["excluded"]), 1)
            self.assertIn("synthetic", result["excluded"][0]["reason"])
            with (output / "comparison.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                comparison = list(csv.DictReader(handle))
            with (output / "per_class_metrics.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                per_class = list(csv.DictReader(handle))
            with (output / "confusion_matrix.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                confusion = list(csv.DictReader(handle))
            self.assertEqual(len(comparison), 1)
            self.assertEqual(comparison[0]["algorithm"], "centralized")
            self.assertEqual(len(per_class), 10)
            self.assertEqual(len(confusion), 100)
            manifest = json.loads(
                (output / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["research_candidates_only"])
            self.assertIn("environment.json", manifest["outputs"])

    def test_export_refuses_to_overwrite_non_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "export"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_results([root / "missing"], output, project_root=root)

    def test_export_excludes_flower_pilot_from_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flower = root / "flower-pilot"
            flower.mkdir()
            (flower / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "result_kind": "federated_image_pilot",
                        "research_result_valid": False,
                    }
                ),
                encoding="utf-8",
            )
            (flower / "metrics.json").write_text("{}", encoding="utf-8")

            result = export_results(
                [flower],
                root / "export",
                project_root=root,
            )

            self.assertEqual(result["included"], [])
            self.assertEqual(len(result["excluded"]), 1)
            self.assertIn(
                "research_result_valid=false",
                result["excluded"][0]["reason"],
            )


if __name__ == "__main__":
    unittest.main()
