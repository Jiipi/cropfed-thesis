"""Central coordinator: orchestration and aggregation, never raw image ingestion."""

from __future__ import annotations

import json
import time
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from cropfed.constants import taxonomy_from_scope
from cropfed.data.torch_data import build_dataloader
from cropfed.experiments.artifacts import (
    file_sha256,
    prepare_new_output_directory,
    write_environment_artifact,
)
from cropfed.experiments.protocol import validate_protocol_lock
from cropfed.flower.tracking import (
    TrackedFedAvg,
    TrackedFedBN,
    TrackedFedProx,
    TrackedMOON,
    TrackedSCAFFOLD,
)
from cropfed.ml.checkpoint import save_model_checkpoint
from cropfed.ml.model import build_model
from cropfed.ml.reporting import flower_evaluation_values
from cropfed.ml.trainer import evaluate_model, select_device, set_reproducible_seed

app = ServerApp()


def _build_strategy(context: Context):
    common = {
        "fraction_train": float(context.run_config["fraction-train"]),
        "fraction_evaluate": float(context.run_config["fraction-evaluate"]),
        "min_train_nodes": int(context.run_config["num-clients"]),
        "min_evaluate_nodes": int(context.run_config["num-clients"]),
        "min_available_nodes": int(context.run_config["num-clients"]),
        "weighted_by_key": "num-examples",
    }
    algorithm = str(context.run_config["algorithm"]).lower()
    if algorithm == "fedavg":
        return TrackedFedAvg(**common)
    if algorithm == "fedprox":
        return TrackedFedProx(
            **common,
            proximal_mu=float(context.run_config["proximal-mu"]),
        )
    if algorithm == "fedbn":
        return TrackedFedBN(**common)
    if algorithm == "scaffold":
        return TrackedSCAFFOLD(
            **common,
            scaffold_server_lr=float(
                context.run_config.get("scaffold-server-lr", 1.0)
            ),
        )
    if algorithm == "moon":
        return TrackedMOON(
            **common,
            moon_temperature=float(
                context.run_config.get("moon-temperature", 0.5)
            ),
            moon_mu=float(context.run_config.get("moon-mu", 1.0)),
        )
    raise ValueError(
        "algorithm must be 'fedavg', 'fedprox', 'fedbn', 'scaffold', or 'moon'"
    )


def _algorithm_hyperparameters(context: Context) -> dict[str, float]:
    """Return every algorithm hyperparameter, zero when not applicable.

    Recorded unconditionally in the checkpoint and run manifest: without them a
    finished ``scaffold`` or ``moon`` artifact cannot be told apart from one run
    at different settings, which makes the results table unreproducible.
    """

    algorithm = str(context.run_config["algorithm"]).lower()
    return {
        "proximal_mu": (
            float(context.run_config["proximal-mu"]) if algorithm == "fedprox" else 0.0
        ),
        "scaffold_server_lr": (
            float(context.run_config.get("scaffold-server-lr", 1.0))
            if algorithm == "scaffold"
            else 0.0
        ),
        "moon_temperature": (
            float(context.run_config.get("moon-temperature", 0.5))
            if algorithm == "moon"
            else 0.0
        ),
        "moon_mu": (
            float(context.run_config.get("moon-mu", 1.0)) if algorithm == "moon" else 0.0
        ),
    }


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run the configured cross-silo federation."""

    seed = int(context.run_config["seed"])
    scope = context.run_config.get("taxonomy-scope")
    if not scope:
        raise KeyError(
            "run_config is missing 'taxonomy-scope'; set it in "
            "[tool.flwr.app.config] or pass it from the run launcher"
        )
    taxonomy = taxonomy_from_scope(str(scope))
    save_model = bool(context.run_config["save-model"])
    configured_result_kind = str(
        context.run_config.get("result-kind", "federated_image_unclassified")
    )
    research_result_valid = bool(
        context.run_config.get("research-result-valid", False)
    )
    result_kind = (
        "federated_image_research_candidate"
        if research_result_valid
        else configured_result_kind
    )
    global_test_manifest = Path(
        str(
            context.run_config.get(
                "global-test-manifest",
                context.run_config.get("central-test-manifest", ""),
            )
        )
    )
    if not global_test_manifest.is_file():
        raise FileNotFoundError(
            f"global test manifest does not exist: {global_test_manifest}"
        )
    algorithm = str(context.run_config["algorithm"]).lower()
    partition_summary = (
        Path(str(context.run_config["client-data-root"])) / "partition_summary.json"
    )
    protocol_validation = None
    if research_result_valid:
        lock_value = str(context.run_config.get("protocol-lock", "")).strip()
        protocol_validation = validate_protocol_lock(
            Path(lock_value) if lock_value else None,
            experiment_type="federated",
            config={
                "algorithm": algorithm,
                "partition_kind": str(context.run_config["partition-kind"]),
                "dirichlet_alpha": float(context.run_config["dirichlet-alpha"]),
                "model": str(context.run_config["model-name"]),
                "num_clients": int(context.run_config["num-clients"]),
                "num_rounds": int(context.run_config["num-server-rounds"]),
                "local_epochs": int(context.run_config["local-epochs"]),
                "batch_size": int(context.run_config["batch-size"]),
                "learning_rate": float(context.run_config["learning-rate"]),
                "pretrained": bool(context.run_config["pretrained"]),
                **_algorithm_hyperparameters(context),
            },
            manifest_hashes={
                "global_test_sha256": file_sha256(global_test_manifest),
                "partition_summary_sha256": (
                    file_sha256(partition_summary)
                    if partition_summary.is_file()
                    else None
                ),
            },
            seed=seed,
        )
    output_dir = (
        prepare_new_output_directory(Path(str(context.run_config["output-dir"])))
        if save_model
        else None
    )
    set_reproducible_seed(seed)
    model = build_model(
        str(context.run_config["model-name"]),
        num_classes=len(taxonomy.class_names),
        pretrained=bool(context.run_config["pretrained"]),
    )
    initial_arrays = ArrayRecord(model.state_dict())
    strategy = _build_strategy(context)

    strategy_started = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(
            {"lr": float(context.run_config["learning-rate"])}
        ),
        num_rounds=int(context.run_config["num-server-rounds"]),
    )
    strategy_elapsed_seconds = time.perf_counter() - strategy_started

    if strategy.best_state_dict is None or strategy.best_validation_round is None:
        raise RuntimeError(
            "Flower produced no validation-selected checkpoint; "
            "client evaluation must run after every training round"
        )
    model.load_state_dict(strategy.best_state_dict)
    global_test_started = time.perf_counter()
    global_test = _evaluate_manifest(
        context,
        model,
        global_test_manifest,
        class_names=taxonomy.class_names,
        class_groups=taxonomy.class_groups,
    )
    global_test_elapsed_seconds = time.perf_counter() - global_test_started
    global_test_metrics = {
        "global_test_evaluation_seconds": global_test_elapsed_seconds,
        **flower_evaluation_values(
            global_test,
            prefix="global_test",
            detailed=True,
            class_names=taxonomy.class_names,
        ),
    }

    if save_model:
        assert output_dir is not None
        checkpoint_path = output_dir / "global_model.pt"
        checkpoint_info = save_model_checkpoint(
            checkpoint_path,
            model,
            model_name=str(context.run_config["model-name"]),
            metadata={
                "experiment_type": "federated",
                "algorithm": str(context.run_config["algorithm"]),
                "taxonomy_scope": taxonomy.scope,
                "partition_kind": str(context.run_config["partition-kind"]),
                "dirichlet_alpha": float(context.run_config["dirichlet-alpha"]),
                "num_clients": int(context.run_config["num-clients"]),
                "num_rounds": int(context.run_config["num-server-rounds"]),
                "local_epochs": int(context.run_config["local-epochs"]),
                "batch_size": int(context.run_config["batch-size"]),
                "learning_rate": float(context.run_config["learning-rate"]),
                "seed": seed,
                "pretrained": bool(context.run_config["pretrained"]),
                "result_kind": result_kind,
                "research_result_valid": research_result_valid,
                "protocol_lock": protocol_validation,
                "best_validation_round": strategy.best_validation_round,
                **_algorithm_hyperparameters(context),
            },
            class_order=taxonomy.class_names,
        )
        history = _result_history(result)
        client_history = sorted(
            getattr(strategy, "client_history", []),
            key=lambda item: (
                int(item["round"]),
                str(item["phase"]),
                int(item["client_id"]),
            ),
        )
        communication = _communication_summary(history)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "client_history": client_history,
                    "communication": communication,
                    "strategy_elapsed_seconds": strategy_elapsed_seconds,
                    "selection": {
                        "criterion": "federated_validation_macro_f1",
                        "tie_breaker": "federated_validation_loss",
                        "best_round": strategy.best_validation_round,
                        "validation_metrics": strategy.best_validation_metrics,
                    },
                    "global_test": global_test_metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        client_metrics_path = output_dir / "client_metrics.json"
        client_metrics_path.write_text(
            json.dumps(client_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        environment = write_environment_artifact(output_dir)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "result_kind": result_kind,
                    "research_result_valid": research_result_valid,
                    "protocol_lock": protocol_validation,
                    "research_validation_status": (
                        "validated_research_candidate"
                        if research_result_valid
                        else "pilot_not_for_research"
                    ),
                    "algorithm": str(context.run_config["algorithm"]),
                    "partition_kind": str(context.run_config["partition-kind"]),
                    "dirichlet_alpha": float(context.run_config["dirichlet-alpha"]),
                    "num_clients": int(context.run_config["num-clients"]),
                    "num_rounds": int(context.run_config["num-server-rounds"]),
                    "local_epochs": int(context.run_config["local-epochs"]),
                    "batch_size": int(context.run_config["batch-size"]),
                    "learning_rate": float(context.run_config["learning-rate"]),
                    "seed": seed,
                    "pretrained": bool(context.run_config["pretrained"]),
                    **_algorithm_hyperparameters(context),
                    "taxonomy_scope": taxonomy.scope,
                    "class_order": list(taxonomy.class_names),
                    "model": str(context.run_config["model-name"]),
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "checkpoint_bytes": checkpoint_info["bytes"],
                    "checkpoint_format_version": checkpoint_info["format_version"],
                    "model_version": checkpoint_info["model_version"],
                    "metrics": metrics_path.name,
                    "client_metrics": client_metrics_path.name,
                    "environment": environment,
                    "history_entries": len(history),
                    "client_history_entries": len(client_history),
                    "communication": communication,
                    "strategy_elapsed_seconds": strategy_elapsed_seconds,
                    "best_validation_round": strategy.best_validation_round,
                    "validation_selection": {
                        "criterion": "federated_validation_macro_f1",
                        "tie_breaker": "federated_validation_loss",
                        "metrics": strategy.best_validation_metrics,
                    },
                    "global_test_evaluated_once_after_selection": True,
                    "global_test_metrics": global_test_metrics,
                    "raw_images_received_by_server": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _result_history(result) -> list[dict[str, object]]:
    rounds = sorted(
        set(result.train_metrics_clientapp)
        | set(result.evaluate_metrics_clientapp)
        | set(result.evaluate_metrics_serverapp)
    )
    history = [
        {
            "round": round_number,
            "train": dict(result.train_metrics_clientapp.get(round_number, {})),
            "federated_evaluate": dict(
                result.evaluate_metrics_clientapp.get(round_number, {})
            ),
            "central_evaluate": dict(
                result.evaluate_metrics_serverapp.get(round_number, {})
            ),
        }
        for round_number in rounds
    ]
    for item in history:
        train = item["train"]
        evaluate = item["federated_evaluate"]
        assert isinstance(train, dict) and isinstance(evaluate, dict)
        item["communication"] = _round_communication(train, evaluate)
        central = item["central_evaluate"]
        assert isinstance(central, dict)
        train_seconds = float(train.get("phase_seconds", 0.0))
        evaluate_seconds = float(evaluate.get("phase_seconds", 0.0))
        central_seconds = float(central.get("central_evaluation_seconds", 0.0))
        item["timing"] = {
            "train_seconds": train_seconds,
            "federated_evaluate_seconds": evaluate_seconds,
            "central_evaluate_seconds": central_seconds,
            "round_seconds": train_seconds + evaluate_seconds + central_seconds,
        }
    return history


def _round_communication(
    train: dict[str, object],
    evaluate: dict[str, object],
) -> dict[str, int]:
    fields = (
        "payload_download_bytes",
        "payload_upload_bytes",
        "model_download_bytes",
        "model_upload_bytes",
    )
    result: dict[str, int] = {}
    for field in fields:
        train_value = int(train.get(f"comm_{field}", 0))
        evaluate_value = int(evaluate.get(f"comm_{field}", 0))
        result[f"train_{field}"] = train_value
        result[f"evaluate_{field}"] = evaluate_value
        result[field] = train_value + evaluate_value
    result["payload_total_bytes"] = (
        result["payload_download_bytes"] + result["payload_upload_bytes"]
    )
    result["model_total_bytes"] = (
        result["model_download_bytes"] + result["model_upload_bytes"]
    )
    return result


def _communication_summary(history: list[dict[str, object]]) -> dict[str, int | str]:
    numeric_fields = (
        "payload_download_bytes",
        "payload_upload_bytes",
        "payload_total_bytes",
        "model_download_bytes",
        "model_upload_bytes",
        "model_total_bytes",
    )
    totals: dict[str, int | str] = {
        "measurement": "flower_record_payload_bytes_excluding_transport_overhead",
    }
    for field in numeric_fields:
        totals[field] = sum(
            int(item.get("communication", {}).get(field, 0))
            for item in history
            if isinstance(item.get("communication"), dict)
        )
    return totals


def _evaluate_manifest(
    context: Context,
    model,
    manifest_path: Path,
    *,
    class_names,
    class_groups,
):
    dataloader = build_dataloader(
        manifest_path,
        training=False,
        batch_size=int(context.run_config["batch-size"]),
    )
    return evaluate_model(
        model,
        dataloader,
        device=select_device(),
        class_names=class_names,
        class_groups=class_groups,
    )
