"""Export research-candidate runs into auditable comparison artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cropfed.constants import PROJECT_VERSION
from cropfed.ml.metrics import client_fairness, gap_vs_centralized

COMPARISON_FIELDS = (
    "run_id",
    "experiment_type",
    "algorithm",
    "model",
    "seed",
    "partition_kind",
    "dirichlet_alpha",
    "num_clients",
    "num_rounds",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "worst_client_macro_f1",
    "best_client_macro_f1",
    "mean_client_macro_f1",
    "client_macro_f1_std",
    "client_macro_f1_spread",
    "gap_vs_centralized_accuracy",
    "gap_vs_centralized_macro_f1",
    "gap_baseline_run_id",
    "harmful_missed_as_healthy_rate",
    "spider_mite_f1",
    "elapsed_seconds",
    "payload_upload_bytes",
    "payload_download_bytes",
    "payload_total_bytes",
    "checkpoint_bytes",
    "source",
)


def export_results(
    run_paths: list[Path],
    output_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Export comparison/per-class/confusion CSVs and an environment manifest."""

    if not run_paths:
        raise ValueError("at least one --run path is required")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []

    for raw_path in run_paths:
        source = raw_path.resolve()
        candidate = _load_candidate(source)
        reason = _exclusion_reason(source, candidate)
        if reason:
            excluded.append({"source": str(source), "reason": reason})
            continue
        normalized = _normalize_candidate(source, candidate)
        comparison_rows.append(normalized["comparison"])
        per_class_rows.extend(normalized["per_class"])
        confusion_rows.extend(normalized["confusion"])
        included.append(
            {
                "source": str(source),
                "run_id": str(normalized["comparison"]["run_id"]),
            }
        )

    _apply_gap_columns(comparison_rows)

    comparison_path = output_dir / "comparison.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    confusion_path = output_dir / "confusion_matrix.csv"
    environment_path = output_dir / "environment.json"
    _write_csv(comparison_path, COMPARISON_FIELDS, comparison_rows)
    _write_csv(
        per_class_path,
        (
            "run_id",
            "client_id",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ),
        per_class_rows,
    )
    _write_csv(
        confusion_path,
        ("run_id", "actual_class_id", "predicted_class_id", "count"),
        confusion_rows,
    )
    environment = build_environment_manifest(project_root or Path.cwd())
    environment_path.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "project_version": PROJECT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "research_candidates_only": True,
        "included": included,
        "excluded": excluded,
        "outputs": {},
    }
    for path in (comparison_path, per_class_path, confusion_path, environment_path):
        manifest["outputs"][path.name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


def build_environment_manifest(project_root: Path) -> dict[str, Any]:
    """Capture the minimum reproducibility environment without leaking secrets."""

    packages: dict[str, str | None] = {}
    for distribution in (
        "cropfed-thesis",
        "flwr",
        "numpy",
        "pillow",
        "torch",
        "torchvision",
        "fastapi",
        "sqlmodel",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None

    cuda_available = False
    cuda_version = None
    device_count = 0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        device_count = int(torch.cuda.device_count())
    except ImportError:
        pass

    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "project_version": PROJECT_VERSION,
        "git_commit": _git_commit(project_root),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda": {
            "available": cuda_available,
            "version": cuda_version,
            "device_count": device_count,
        },
    }


def _load_candidate(path: Path) -> dict[str, Any]:
    if path.is_dir():
        if (path / "run_manifest.json").is_file():
            return {
                "kind": "flower",
                "manifest": _read_json(path / "run_manifest.json"),
                "metrics": _read_json(path / "metrics.json"),
            }
        if (path / "result.json").is_file():
            return {"kind": "baseline", "result": _read_json(path / "result.json")}
        raise FileNotFoundError(f"no supported run artifact in {path}")
    if path.is_file():
        payload = _read_json(path)
        if "algorithm" in payload and "history" in payload:
            return {"kind": "api-or-smoke", "result": payload}
        return {"kind": "baseline", "result": payload}
    raise FileNotFoundError(f"run path does not exist: {path}")


def _exclusion_reason(path: Path, candidate: dict[str, Any]) -> str | None:
    result = candidate.get("result", {})
    manifest = candidate.get("manifest", {})

    if isinstance(result, dict):
        result_kind = str(result.get("result_kind", "")).lower()
        if "synthetic" in result_kind:
            return "synthetic smoke results are forbidden in research exports"
    if isinstance(manifest, dict):
        result_kind = str(manifest.get("result_kind", "")).lower()
        if "synthetic" in result_kind:
            return "synthetic Flower results are forbidden in research exports"

    for ancestor in (path if path.is_dir() else path.parent, *path.parents):
        fixture_path = ancestor / "fixture.json"
        if fixture_path.is_file():
            fixture = _read_json(fixture_path)
            if fixture.get("fixture_kind") == "synthetic_images_for_integration_only":
                return "Flower run belongs to a synthetic image fixture"

    valid_in_result = (
        result.get("research_result_valid") if isinstance(result, dict) else None
    )
    valid_in_manifest = (
        manifest.get("research_result_valid") if isinstance(manifest, dict) else None
    )
    is_valid = valid_in_result if valid_in_result is not None else valid_in_manifest

    if is_valid is False:
        return "run is explicitly marked research_result_valid=false"
    if is_valid is not True:
        return (
            "run is not explicitly validated as a research candidate "
            "(research_result_valid must be True)"
        )
    return None


def _normalize_candidate(path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate["kind"] == "flower":
        return _normalize_flower(path, candidate["manifest"], candidate["metrics"])
    result = candidate["result"]
    experiment_type = str(result.get("experiment_type", ""))
    if experiment_type == "centralized":
        return _normalize_centralized(path, result)
    if experiment_type == "local-only":
        return _normalize_local_only(path, result)
    raise ValueError(f"unsupported research result shape: {path}")


def _normalize_centralized(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    run_id = _run_id(path, "centralized", result.get("seed"))
    comparison = _base_comparison(run_id, path, result)
    comparison.update(_common_metrics(metrics))
    comparison.update(
        {
            "algorithm": "centralized",
            "partition_kind": "pooled",
            "elapsed_seconds": result.get("elapsed_seconds"),
            "checkpoint_bytes": result.get("checkpoint_bytes"),
        }
    )
    return {
        "comparison": comparison,
        "per_class": _per_class_from_rich(
            run_id, metrics, class_order=_require_class_order(result, run_id)
        ),
        "confusion": _confusion_rows(run_id, metrics.get("confusion_matrix", [])),
    }


def _normalize_local_only(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    run_id = _run_id(path, "local-only", result.get("seed"))
    summary = result["summary"]
    clients = [
        client for client in result.get("clients", []) if isinstance(client, dict)
    ]
    client_f1s = [
        float(client["global_test_metrics"]["macro_f1"])
        for client in clients
        if isinstance(client.get("global_test_metrics"), dict)
        and client["global_test_metrics"].get("macro_f1") is not None
    ]
    # Every client is scored on the same global test set (D-024), so the
    # unweighted mean is the fair one; local training size is what tells us
    # whether a low score belongs to a small facility.
    client_sizes = [int(client.get("num_train", 0)) for client in clients]
    comparison = _base_comparison(run_id, path, result)
    comparison.update(
        {
            "algorithm": "local-only",
            "accuracy": summary.get("mean_global_accuracy"),
            "macro_f1": summary.get("mean_global_macro_f1"),
            "worst_client_macro_f1": summary.get("worst_global_macro_f1"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "checkpoint_bytes": sum(
                int(client.get("checkpoint_bytes", 0))
                for client in result.get("clients", [])
            ),
        }
    )
    comparison.update(
        _fairness_columns(
            client_f1s,
            num_examples=(
                client_sizes
                if len(client_sizes) == len(client_f1s) and sum(client_sizes) > 0
                else None
            ),
        )
    )
    per_class: list[dict[str, Any]] = []
    class_order = _require_class_order(result, run_id)
    for client in result.get("clients", []):
        per_class.extend(
            _per_class_from_rich(
                run_id,
                client.get("global_test_metrics", {}),
                client_id=int(client["client_id"]),
                class_order=class_order,
            )
        )
    return {"comparison": comparison, "per_class": per_class, "confusion": []}


def _normalize_flower(
    path: Path,
    manifest: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> dict[str, Any]:
    history = metrics_payload.get("history", [])
    recorded_global_test = metrics_payload.get("global_test")
    if isinstance(recorded_global_test, dict) and recorded_global_test:
        final = recorded_global_test
        prefix = "global_test_"
    else:
        final = next(
            (
                row.get("central_evaluate", {})
                for row in reversed(history)
                if row.get("central_evaluate")
            ),
            {},
        )
        prefix = "central_"
    selection = metrics_payload.get("selection", {})
    selected_round = (
        int(selection["best_round"])
        if isinstance(selection, dict) and selection.get("best_round") is not None
        else None
    )
    algorithm = str(manifest["algorithm"])
    run_id = _run_id(path, algorithm, manifest.get("seed"))
    client_scores, client_sizes = _client_eval_scores(
        metrics_payload.get("client_history", []),
        round_number=selected_round,
    )
    comparison = _base_comparison(run_id, path, manifest)
    comparison.update(
        {
            "experiment_type": "federated",
            "algorithm": algorithm,
            "accuracy": final.get(f"{prefix}accuracy"),
            "macro_precision": final.get(f"{prefix}macro_precision"),
            "macro_recall": final.get(f"{prefix}macro_recall"),
            "macro_f1": final.get(f"{prefix}macro_f1"),
            "worst_client_macro_f1": min(client_scores) if client_scores else None,
            "harmful_missed_as_healthy_rate": final.get(
                f"{prefix}harmful_missed_as_healthy_rate"
            ),
            "spider_mite_f1": final.get(f"{prefix}spider_mite_f1"),
            "elapsed_seconds": metrics_payload.get("strategy_elapsed_seconds"),
            "payload_upload_bytes": metrics_payload.get("communication", {}).get(
                "payload_upload_bytes"
            ),
            "payload_download_bytes": metrics_payload.get("communication", {}).get(
                "payload_download_bytes"
            ),
            "payload_total_bytes": metrics_payload.get("communication", {}).get(
                "payload_total_bytes"
            ),
            "checkpoint_bytes": manifest.get("checkpoint_bytes"),
        }
    )
    comparison.update(
        _fairness_columns(
            client_scores,
            num_examples=client_sizes if sum(client_sizes) > 0 else None,
        )
    )
    class_order = _require_class_order(manifest, run_id)
    per_class = _per_class_from_flat(run_id, final, class_order, prefix)
    size = int(final.get(f"{prefix}confusion_matrix_size", 0))
    flat = final.get(f"{prefix}confusion_matrix_flat", [])
    matrix = [flat[index * size : (index + 1) * size] for index in range(size)]
    return {
        "comparison": comparison,
        "per_class": per_class,
        "confusion": _confusion_rows(run_id, matrix),
    }


def _apply_gap_columns(rows: list[dict[str, Any]]) -> None:
    """Fill the gap-vs-centralized columns in place, once all runs are known.

    §8 calls the distance to centralized training the core result of the
    thesis, so it is a column rather than something a reader subtracts by eye.
    Two rules keep the column from quietly comparing the wrong pair:

    * The baseline is matched on ``(seed, model)``, not on seed alone. A
      federated MobileNetV3 run measured against a centralized EfficientNet
      baseline would report an architecture difference as a federation cost.
    * A run with no recorded seed is never paired. Its gap would assert a
      same-seed comparison that nothing in the artifact supports.
    """

    baselines: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in rows:
        if row.get("algorithm") != "centralized" or row.get("seed") is None:
            continue
        key = (row.get("seed"), row.get("model"))
        existing = baselines.get(key)
        if existing is not None:
            raise ValueError(
                "two centralized baselines share seed "
                f"{key[0]!r} and model {key[1]!r} "
                f"({existing['run_id']} and {row['run_id']}); "
                "export one of them so the gap column names a single baseline"
            )
        baselines[key] = row

    for row in rows:
        if row.get("seed") is None:
            continue
        baseline = baselines.get((row.get("seed"), row.get("model")))
        if baseline is None:
            continue
        row["gap_vs_centralized_accuracy"] = gap_vs_centralized(
            row.get("accuracy"), baseline.get("accuracy")
        )
        row["gap_vs_centralized_macro_f1"] = gap_vs_centralized(
            row.get("macro_f1"), baseline.get("macro_f1")
        )
        row["gap_baseline_run_id"] = baseline["run_id"]


def _fairness_columns(
    scores: list[float],
    *,
    num_examples: list[int] | None = None,
) -> dict[str, Any]:
    """Map ``client_fairness`` onto comparison columns, or blanks when absent.

    A single client is not a federation, so the spread of one score is left
    blank rather than reported as a perfectly fair 0.0.
    """

    if len(scores) < 2:
        return {
            "best_client_macro_f1": scores[0] if scores else None,
            "mean_client_macro_f1": scores[0] if scores else None,
            "client_macro_f1_std": None,
            "client_macro_f1_spread": None,
        }
    fairness = client_fairness(scores, num_examples=num_examples)
    return {
        "best_client_macro_f1": fairness["best"],
        "mean_client_macro_f1": fairness["mean"],
        "client_macro_f1_std": fairness["std"],
        "client_macro_f1_spread": fairness["spread"],
    }


def _base_comparison(
    run_id: str, path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        field: None
        for field in COMPARISON_FIELDS
    } | {
        "run_id": run_id,
        "experiment_type": payload.get("experiment_type"),
        "model": payload.get("model"),
        "seed": payload.get("seed"),
        "partition_kind": payload.get("partition_kind"),
        "dirichlet_alpha": payload.get("dirichlet_alpha"),
        "num_clients": payload.get("num_clients"),
        "num_rounds": payload.get("num_rounds"),
        "source": str(path),
    }


def _common_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = metrics.get("per_class", {})
    spider = next(
        (
            value
            for name, value in per_class.items()
            if "spider mite" in name.lower()
        ),
        {},
    )
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "harmful_missed_as_healthy_rate": metrics.get(
            "harmful_missed_as_healthy_rate"
        ),
        "spider_mite_f1": spider.get("f1"),
    }


def _require_class_order(
    source: dict[str, Any], run_id: str
) -> list[str]:
    """Return the run's own class order, refusing to guess it.

    Substituting a default taxonomy here would silently mislabel every
    per-class row of a run trained on a different taxonomy, which is worse
    than failing the export.
    """

    class_order = source.get("class_order")
    if not class_order:
        raise ValueError(
            f"run {run_id!r} has no 'class_order'; re-export it from a "
            "checkpoint that records its taxonomy instead of assuming one"
        )
    return list(class_order)


def _per_class_from_rich(
    run_id: str,
    metrics: dict[str, Any],
    *,
    client_id: int | None = None,
    class_order: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    per_class = metrics.get("per_class", {})
    return [
        {
            "run_id": run_id,
            "client_id": client_id,
            "class_id": class_id,
            "class_name": name,
            "precision": per_class.get(name, {}).get("precision"),
            "recall": per_class.get(name, {}).get("recall"),
            "f1": per_class.get(name, {}).get("f1"),
            "support": per_class.get(name, {}).get("support"),
        }
        for class_id, name in enumerate(class_order)
    ]


def _per_class_from_flat(
    run_id: str,
    metrics: dict[str, Any],
    class_order: list[str],
    prefix: str,
) -> list[dict[str, Any]]:
    precision = metrics.get(f"{prefix}per_class_precision", [])
    recall = metrics.get(f"{prefix}per_class_recall", [])
    f1 = metrics.get(f"{prefix}per_class_f1", [])
    support = metrics.get(f"{prefix}per_class_support", [])
    return [
        {
            "run_id": run_id,
            "client_id": None,
            "class_id": class_id,
            "class_name": name,
            "precision": _list_value(precision, class_id),
            "recall": _list_value(recall, class_id),
            "f1": _list_value(f1, class_id),
            "support": _list_value(support, class_id),
        }
        for class_id, name in enumerate(class_order)
    ]


def _confusion_rows(run_id: str, matrix: list[list[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "actual_class_id": actual,
            "predicted_class_id": predicted,
            "count": value,
        }
        for actual, row in enumerate(matrix)
        for predicted, value in enumerate(row)
    ]


def _client_eval_scores(
    client_history: object,
    *,
    round_number: int | None = None,
) -> tuple[list[float], list[int]]:
    """Return each client's macro-F1 at the selected round, with its test size.

    Scores are read from one round only. Pooling rounds would average a
    client's early bad rounds into its final score and make the federation
    look more uniform than it is at the checkpoint actually being shipped.
    """

    if not isinstance(client_history, list):
        return [], []
    selected_round = round_number
    if selected_round is None:
        selected_round = max(
            (int(item["round"]) for item in client_history if isinstance(item, dict)),
            default=0,
        )
    entries = [
        item
        for item in client_history
        if isinstance(item, dict)
        and item.get("phase") == "evaluate"
        and int(item.get("round", 0)) == selected_round
        and item.get("metrics", {}).get("eval_macro_f1") is not None
    ]
    entries.sort(key=lambda item: int(item.get("client_id", 0)))
    scores = [float(item["metrics"]["eval_macro_f1"]) for item in entries]
    sizes = [int(item.get("num_examples", 0)) for item in entries]
    return scores, sizes


def _run_id(path: Path, scenario: str, seed: object) -> str:
    source_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{scenario}-seed-{seed}-{source_hash}"


def _list_value(values: object, index: int) -> object:
    return values[index] if isinstance(values, list) and index < len(values) else None


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
