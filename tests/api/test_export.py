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
                        "research_result_valid": True,
                        "metrics": metrics,
                        "class_order": list(TOMATO_CLASSES),
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

    def test_flower_export_uses_one_shot_global_test_and_selected_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flower = root / "flower-main"
            flower.mkdir()
            (flower / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "result_kind": "federated_image_research_candidate",
                        "research_result_valid": True,
                        "experiment_type": "federated",
                        "algorithm": "fedavg",
                        "model": "mobilenet_v2",
                        "seed": 2026,
                        "partition_kind": "iid",
                        "num_clients": 4,
                        "num_rounds": 2,
                        "class_order": list(TOMATO_CLASSES),
                    }
                ),
                encoding="utf-8",
            )
            (flower / "metrics.json").write_text(
                json.dumps(
                    {
                        "history": [
                            {"round": 1, "federated_evaluate": {"eval_macro_f1": 0.7}},
                            {"round": 2, "federated_evaluate": {"eval_macro_f1": 0.6}},
                        ],
                        "selection": {"best_round": 1},
                        "global_test": {
                            "global_test_accuracy": 0.85,
                            "global_test_macro_precision": 0.81,
                            "global_test_macro_recall": 0.82,
                            "global_test_macro_f1": 0.83,
                            "global_test_harmful_missed_as_healthy_rate": 0.05,
                            "global_test_spider_mite_f1": 0.77,
                            "global_test_confusion_matrix_size": 0,
                        },
                        "client_history": [
                            {
                                "round": 1,
                                "phase": "evaluate",
                                "client_id": client_id,
                                "metrics": {"eval_macro_f1": 0.2 + client_id * 0.1},
                            }
                            for client_id in range(4)
                        ]
                        + [
                            {
                                "round": 2,
                                "phase": "evaluate",
                                "client_id": client_id,
                                "metrics": {"eval_macro_f1": 0.9},
                            }
                            for client_id in range(4)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            export_results([flower], root / "export", project_root=root)
            with (root / "export" / "comparison.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                comparison = next(csv.DictReader(handle))

            self.assertEqual(float(comparison["macro_f1"]), 0.83)
            self.assertEqual(float(comparison["worst_client_macro_f1"]), 0.2)


if __name__ == "__main__":
    unittest.main()
