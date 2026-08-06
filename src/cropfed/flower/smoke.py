"""Validation helpers for real Flower runs over synthetic image fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cropfed.ml.checkpoint import load_model_checkpoint

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Remove terminal control sequences before logs are persisted or parsed."""

    return _ANSI_ESCAPE.sub("", text)


def parse_flower_log_evidence(
    log_text: str,
    *,
    algorithm: str,
    expected_clients: int,
    proximal_mu: float,
) -> dict[str, Any]:
    """Extract strict integration evidence from Flower's human-readable log."""

    normalized = strip_ansi(log_text)
    algorithm = algorithm.lower()
    if algorithm not in {"fedavg", "fedprox"}:
        raise ValueError("algorithm must be 'fedavg' or 'fedprox'")

    strategy_names = (
        ("FedAvg", "TrackedFedAvg")
        if algorithm == "fedavg"
        else ("FedProx", "TrackedFedProx")
    )
    strategy_started = any(
        f"Starting {strategy_name} strategy:" in normalized
        for strategy_name in strategy_names
    )
    train_pattern = (
        rf"aggregate_train: Received {expected_clients} results and 0 failures"
    )
    evaluate_pattern = (
        rf"aggregate_evaluate: Received {expected_clients} results and 0 failures"
    )
    evidence: dict[str, Any] = {
        "strategy_started": strategy_started,
        "registered_clients": (
            f"Registered {expected_clients} nodes" in normalized
            or f"({expected_clients} simulated SuperNodes)" in normalized
        ),
        "train_results_complete": re.search(train_pattern, normalized) is not None,
        "evaluate_results_complete": re.search(evaluate_pattern, normalized) is not None,
    }
    if algorithm == "fedprox":
        proximal_pattern = rf"Proximal mu:\s+{re.escape(str(proximal_mu))}(?:\s|$)"
        evidence["proximal_mu_confirmed"] = (
            re.search(proximal_pattern, normalized) is not None
        )

    failed = [name for name, passed in evidence.items() if passed is not True]
    if failed:
        raise RuntimeError(
            "Flower log is missing required integration evidence: " + ", ".join(failed)
        )
    return evidence


def validate_run_artifacts(
    output_dir: Path,
    *,
    algorithm: str,
    expected_clients: int,
    proximal_mu: float,
    log_text: str,
    expected_class_order: Sequence[str],
) -> dict[str, Any]:
    """Validate checkpoint, manifest, taxonomy and 4-client Flower evidence."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Flower run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    algorithm = algorithm.lower()
    expected = {
        "algorithm": algorithm,
        "num_clients": expected_clients,
        "raw_images_received_by_server": False,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Flower run manifest mismatch: {mismatches}")
    if algorithm == "fedprox" and float(manifest.get("proximal_mu", -1.0)) != proximal_mu:
        raise RuntimeError("Flower run manifest has the wrong FedProx proximal_mu")
    resolved_class_order = tuple(expected_class_order)
    if tuple(manifest.get("class_order", resolved_class_order)) != resolved_class_order:
        raise RuntimeError("Flower run manifest has the wrong class order")

    checkpoint_value = Path(str(manifest["checkpoint"]))
    checkpoint_path = (
        checkpoint_value
        if checkpoint_value.is_absolute()
        else output_dir / checkpoint_value
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Flower checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if checkpoint_sha256 != manifest.get("checkpoint_sha256"):
        raise RuntimeError("Flower checkpoint SHA-256 does not match its run manifest")
    if checkpoint_path.stat().st_size != int(manifest.get("checkpoint_bytes", -1)):
        raise RuntimeError("Flower checkpoint size does not match its run manifest")

    loaded = load_model_checkpoint(checkpoint_path)
    if loaded.class_order != resolved_class_order:
        raise RuntimeError("checkpoint class order does not match the run manifest")
    if loaded.metadata.get("algorithm") != algorithm:
        raise RuntimeError("checkpoint metadata algorithm does not match the run")
    if loaded.metadata.get("num_clients") != expected_clients:
        raise RuntimeError("checkpoint metadata client count does not match the run")

    metrics_path = _artifact_path(output_dir, manifest, "metrics")
    client_metrics_path = _artifact_path(output_dir, manifest, "client_metrics")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeError("Flower run manifest is missing environment metadata")
    environment_path = output_dir / str(environment.get("path", ""))
    if not environment_path.is_file():
        raise FileNotFoundError("Flower environment artifact not found")
    if _sha256_file(environment_path) != environment.get("sha256"):
        raise RuntimeError("Flower environment SHA-256 does not match its run manifest")
    if environment_path.stat().st_size != int(environment.get("bytes", -1)):
        raise RuntimeError("Flower environment size does not match its run manifest")
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    num_rounds = int(manifest.get("num_rounds", 0))
    selection = metrics_payload.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("Flower metrics artifact is missing validation selection")
    best_round = selection.get("best_round")
    if (
        isinstance(best_round, bool)
        or not isinstance(best_round, int)
        or not 1 <= best_round <= num_rounds
    ):
        raise RuntimeError("Flower validation-selected round is invalid")
    if loaded.metadata.get("best_validation_round") != best_round:
        raise RuntimeError("checkpoint best round does not match validation selection")
    global_test = metrics_payload.get("global_test")
    if not isinstance(global_test, dict) or not isinstance(
        global_test.get("global_test_macro_f1"), int | float
    ):
        raise RuntimeError("Flower metrics artifact is missing final global test")
    if manifest.get("global_test_evaluated_once_after_selection") is not True:
        raise RuntimeError("Flower run manifest does not confirm one-shot global test")
    client_history = metrics_payload.get("client_history")
    if not isinstance(client_history, list):
        raise RuntimeError("Flower metrics artifact is missing client_history")
    expected_history_entries = expected_clients * num_rounds * 2
    identities = {
        (int(item["round"]), int(item["client_id"]), str(item["phase"]))
        for item in client_history
        if isinstance(item, dict)
    }
    expected_identities = {
        (round_number, client_id, phase)
        for round_number in range(1, num_rounds + 1)
        for client_id in range(expected_clients)
        for phase in ("train", "evaluate")
    }
    if len(client_history) != expected_history_entries or identities != expected_identities:
        raise RuntimeError("Flower client_history is incomplete")
    separate_client_history = json.loads(
        client_metrics_path.read_text(encoding="utf-8")
    )
    if separate_client_history != client_history:
        raise RuntimeError("client_metrics.json does not match metrics.json")
    communication = metrics_payload.get("communication")
    if not isinstance(communication, dict):
        raise RuntimeError("Flower metrics artifact is missing communication summary")
    if int(communication.get("payload_total_bytes", 0)) <= 0:
        raise RuntimeError("Flower communication payload byte count is not positive")

    normalized_log = strip_ansi(log_text)
    evidence = parse_flower_log_evidence(
        normalized_log,
        algorithm=algorithm,
        expected_clients=expected_clients,
        proximal_mu=proximal_mu,
    )
    warnings: list[str] = []
    if "Windows fatal exception" in normalized_log or "access violation" in normalized_log:
        warnings.append("ray_windows_worker_access_violation_logged")
    if "Installing application dependencies" in normalized_log:
        warnings.append("flower_runtime_dependency_environment_created")

    return {
        "status": "passed",
        "algorithm": algorithm,
        "num_clients": expected_clients,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_format_version": loaded.format_version,
        "model_version": loaded.model_version,
        "class_count": len(loaded.class_order or ()),
        "client_history_entries": len(client_history),
        "communication": communication,
        "raw_images_received_by_server": False,
        "evidence": evidence,
        "warnings": warnings,
    }


def _artifact_path(output_dir: Path, manifest: dict[str, Any], key: str) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Flower run manifest is missing {key}")
    path = Path(value)
    resolved = path if path.is_absolute() else output_dir / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Flower artifact not found: {resolved}")
    return resolved


def compare_checkpoint_states(
    first_path: Path,
    second_path: Path,
) -> dict[str, int | float]:
    """Measure actual tensor differences, excluding checkpoint envelope metadata."""

    import torch

    first = load_model_checkpoint(first_path).state_dict
    second = load_model_checkpoint(second_path).state_dict
    if set(first) != set(second):
        raise RuntimeError("checkpoint state dict keys do not match")

    different_tensors = 0
    different_values = 0
    squared_l2 = 0.0
    max_abs = 0.0
    with torch.no_grad():
        for name, first_tensor in first.items():
            second_tensor = second[name]
            if first_tensor.shape != second_tensor.shape:
                raise RuntimeError(f"checkpoint tensor shape mismatch: {name}")
            difference = (
                first_tensor.detach().cpu().double()
                - second_tensor.detach().cpu().double()
            )
            nonzero = int(torch.count_nonzero(difference).item())
            if nonzero:
                different_tensors += 1
                different_values += nonzero
                max_abs = max(max_abs, float(torch.max(torch.abs(difference)).item()))
                squared_l2 += float(torch.sum(difference * difference).item())
    return {
        "tensor_count": len(first),
        "different_tensors": different_tensors,
        "different_values": different_values,
        "max_abs_difference": max_abs,
        "l2_distance": squared_l2**0.5,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
