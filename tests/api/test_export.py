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


def _centralized_result(
    *,
    seed: int,
    model: str = "mobilenet_v2",
    accuracy: float,
    macro_f1: float,
) -> dict[str, object]:
    """A centralized artifact with the two metrics the gap column reads."""

    metrics = classification_metrics(
        list(range(10)),
        list(range(10)),
        num_classes=10,
        class_names=TOMATO_CLASSES,
        healthy_class_id=0,
        class_groups=["healthy", *("disease" for _ in range(8)), "pest"],
    )
    metrics["accuracy"] = accuracy
    metrics["macro_f1"] = macro_f1
    return {
        "experiment_type": "centralized",
        "model": model,
        "seed": seed,
        "research_result_valid": True,
        "metrics": metrics,
        "class_order": list(TOMATO_CLASSES),
        "elapsed_seconds": 1.0,
        "checkpoint_bytes": 100,
    }


def _flower_run(
    directory: Path,
    *,
    seed: int,
    model: str = "mobilenet_v2",
    accuracy: float,
    macro_f1: float,
    client_f1s: list[float],
    client_examples: list[int] | None = None,
) -> Path:
    directory.mkdir(parents=True)
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "result_kind": "federated_image_research_candidate",
                "research_result_valid": True,
                "experiment_type": "federated",
                "algorithm": "fedavg",
                "model": model,
                "seed": seed,
                "partition_kind": "iid",
                "num_clients": len(client_f1s),
                "num_rounds": 1,
                "class_order": list(TOMATO_CLASSES),
            }
        ),
        encoding="utf-8",
    )
    sizes = client_examples or [100] * len(client_f1s)
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "history": [],
                "selection": {"best_round": 1},
                "global_test": {
                    "global_test_accuracy": accuracy,
                    "global_test_macro_f1": macro_f1,
                    "global_test_confusion_matrix_size": 0,
                },
                "client_history": [
                    {
                        "round": 1,
                        "phase": "evaluate",
                        "client_id": client_id,
                        "num_examples": sizes[client_id],
                        "metrics": {"eval_macro_f1": score},
                    }
                    for client_id, score in enumerate(client_f1s)
                ],
            }
        ),
        encoding="utf-8",
    )
    return directory


def _read_comparison(output: Path) -> dict[str, dict[str, str]]:
    with (output / "comparison.csv").open(newline="", encoding="utf-8-sig") as handle:
        return {row["algorithm"]: row for row in csv.DictReader(handle)}


class FairnessAndGapColumnTests(unittest.TestCase):
    def test_flower_export_reports_client_spread_not_only_the_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _flower_run(
                root / "fedavg",
                seed=2026,
                accuracy=0.80,
                macro_f1=0.70,
                client_f1s=[0.4, 0.6, 0.8, 1.0],
            )

            export_results([run], root / "export", project_root=root)
            row = _read_comparison(root / "export")["fedavg"]

            self.assertAlmostEqual(float(row["worst_client_macro_f1"]), 0.4)
            self.assertAlmostEqual(float(row["best_client_macro_f1"]), 1.0)
            self.assertAlmostEqual(float(row["mean_client_macro_f1"]), 0.7)
            self.assertAlmostEqual(float(row["client_macro_f1_spread"]), 0.6)
            self.assertAlmostEqual(
                float(row["client_macro_f1_std"]), 0.223606797749979
            )

    def test_gap_column_is_positive_when_federation_trails_centralized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centralized = root / "centralized"
            centralized.mkdir()
            (centralized / "result.json").write_text(
                json.dumps(
                    _centralized_result(seed=2026, accuracy=0.90, macro_f1=0.88)
                ),
                encoding="utf-8",
            )
            federated = _flower_run(
                root / "fedavg",
                seed=2026,
                accuracy=0.85,
                macro_f1=0.83,
                client_f1s=[0.8, 0.86],
            )

            export_results(
                [centralized, federated], root / "export", project_root=root
            )
            rows = _read_comparison(root / "export")

            self.assertAlmostEqual(
                float(rows["fedavg"]["gap_vs_centralized_accuracy"]), 0.05
            )
            self.assertAlmostEqual(
                float(rows["fedavg"]["gap_vs_centralized_macro_f1"]), 0.05
            )
            self.assertEqual(
                rows["fedavg"]["gap_baseline_run_id"], rows["centralized"]["run_id"]
            )
            # The baseline's gap against itself is 0.0, which is true and worth
            # showing: it is the row a reader checks the column's sign against.
            self.assertAlmostEqual(
                float(rows["centralized"]["gap_vs_centralized_macro_f1"]), 0.0
            )

    def test_gap_is_blank_rather_than_computed_across_different_models(self) -> None:
        """An architecture difference must not be reported as a federation cost."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centralized = root / "centralized"
            centralized.mkdir()
            (centralized / "result.json").write_text(
                json.dumps(
                    _centralized_result(
                        seed=2026,
                        model="mobilenet_v2",
                        accuracy=0.90,
                        macro_f1=0.88,
                    )
                ),
                encoding="utf-8",
            )
            federated = _flower_run(
                root / "fedavg",
                seed=2026,
                model="mobilenet_v3_small",
                accuracy=0.85,
                macro_f1=0.83,
                client_f1s=[0.8, 0.86],
            )

            export_results(
                [centralized, federated], root / "export", project_root=root
            )
            row = _read_comparison(root / "export")["fedavg"]

            self.assertEqual(row["gap_vs_centralized_macro_f1"], "")
            self.assertEqual(row["gap_baseline_run_id"], "")

    def test_gap_is_blank_across_different_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centralized = root / "centralized"
            centralized.mkdir()
            (centralized / "result.json").write_text(
                json.dumps(
                    _centralized_result(seed=2026, accuracy=0.90, macro_f1=0.88)
                ),
                encoding="utf-8",
            )
            federated = _flower_run(
                root / "fedavg",
                seed=2027,
                accuracy=0.85,
                macro_f1=0.83,
                client_f1s=[0.8, 0.86],
            )

            export_results(
                [centralized, federated], root / "export", project_root=root
            )
            row = _read_comparison(root / "export")["fedavg"]

            self.assertEqual(row["gap_vs_centralized_macro_f1"], "")

    def test_gap_is_blank_when_no_centralized_baseline_was_exported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            federated = _flower_run(
                root / "fedavg",
                seed=2026,
                accuracy=0.85,
                macro_f1=0.83,
                client_f1s=[0.8, 0.86],
            )

            export_results([federated], root / "export", project_root=root)
            row = _read_comparison(root / "export")["fedavg"]

            self.assertEqual(row["gap_vs_centralized_accuracy"], "")
            self.assertEqual(row["gap_vs_centralized_macro_f1"], "")
            self.assertEqual(row["gap_baseline_run_id"], "")

    def test_two_baselines_for_one_seed_and_model_is_refused(self) -> None:
        """Silently picking one would make the gap column non-reproducible."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("centralized-a", "centralized-b"):
                directory = root / name
                directory.mkdir()
                (directory / "result.json").write_text(
                    json.dumps(
                        _centralized_result(seed=2026, accuracy=0.90, macro_f1=0.88)
                    ),
                    encoding="utf-8",
                )

            with self.assertRaises(ValueError) as caught:
                export_results(
                    [root / "centralized-a", root / "centralized-b"],
                    root / "export",
                    project_root=root,
                )

            self.assertIn("share seed", str(caught.exception))

    def test_single_client_run_leaves_spread_blank_not_zero(self) -> None:
        """One client is not a federation; 0.0 spread would claim perfect fairness."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _flower_run(
                root / "fedavg",
                seed=2026,
                accuracy=0.80,
                macro_f1=0.70,
                client_f1s=[0.7],
            )

            export_results([run], root / "export", project_root=root)
            row = _read_comparison(root / "export")["fedavg"]

            self.assertEqual(row["client_macro_f1_std"], "")
            self.assertEqual(row["client_macro_f1_spread"], "")
            self.assertAlmostEqual(float(row["worst_client_macro_f1"]), 0.7)

    def test_local_only_export_reports_client_spread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = classification_metrics(
                list(range(10)),
                list(range(10)),
                num_classes=10,
                class_names=TOMATO_CLASSES,
                healthy_class_id=0,
                class_groups=["healthy", *("disease" for _ in range(8)), "pest"],
            )
            local = root / "local-only"
            local.mkdir()
            (local / "result.json").write_text(
                json.dumps(
                    {
                        "experiment_type": "local-only",
                        "model": "mobilenet_v2",
                        "seed": 2026,
                        "research_result_valid": True,
                        "class_order": list(TOMATO_CLASSES),
                        "summary": {
                            "mean_global_accuracy": 0.5,
                            "mean_global_macro_f1": 0.5,
                            "worst_global_macro_f1": 0.2,
                            "elapsed_seconds": 3.0,
                        },
                        "clients": [
                            {
                                "client_id": client_id,
                                "num_train": 100,
                                "checkpoint_bytes": 10,
                                "global_test_metrics": {
                                    **metrics,
                                    "macro_f1": score,
                                },
                            }
                            for client_id, score in enumerate([0.2, 0.4, 0.6, 0.8])
                        ],
                    }
                ),
                encoding="utf-8",
            )

            export_results([local], root / "export", project_root=root)
            row = _read_comparison(root / "export")["local-only"]

            self.assertAlmostEqual(float(row["client_macro_f1_spread"]), 0.6)
            self.assertAlmostEqual(float(row["mean_client_macro_f1"]), 0.5)
            self.assertAlmostEqual(
                float(row["client_macro_f1_std"]), 0.223606797749979
            )


if __name__ == "__main__":
    unittest.main()
