"""Agricultural client: local image access, training and evaluation only."""

from __future__ import annotations

from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from cropfed.constants import taxonomy_from_scope
from cropfed.data.torch_data import build_dataloader
from cropfed.fl.aggregation import batch_norm_parameter_names
from cropfed.flower.tracking import SCAFFOLD_C_DELTA_RECORD, SCAFFOLD_C_RECORD
from cropfed.ml.model import build_model
from cropfed.ml.reporting import flower_evaluation_values
from cropfed.ml.trainer import (
    evaluate_model,
    select_device,
    set_reproducible_seed,
    train_local,
)

app = ClientApp()

#: Client-local state carried between rounds by the Flower runtime.
MOON_PREVIOUS_MODEL_RECORD = "moon_previous_model"
SCAFFOLD_CLIENT_C_RECORD = "scaffold_client_c"
#: FedBN: this client's own batch-norm statistics, never sent to the server.
FEDBN_LOCAL_BN_RECORD = "fedbn_local_batch_norm"


def _client_manifest(context: Context, filename: str) -> Path:
    partition_id = _partition_id(context)
    root = Path(str(context.run_config["client-data-root"]))
    path = root / f"client_{partition_id}" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"client {partition_id} cannot access its local manifest: {path}"
        )
    return path


def _partition_id(context: Context) -> int:
    return int(context.node_config["partition-id"])


def _dataset_root(context: Context) -> Path:
    """Return the anchor for this run's relative image paths.

    Missing is an error, not a default: resolving relative manifest paths against
    the launch directory would make the run depend on where it was started from,
    and every client would fail on its first image instead of saying why.
    """

    value = str(context.run_config.get("dataset-root", "")).strip()
    if not value:
        raise KeyError(
            "run_config is missing 'dataset-root'; set it in "
            "[tool.flwr.app.config] or pass it from the run launcher"
        )
    return Path(value).expanduser().resolve()


def _num_workers(context: Context) -> int:
    return int(context.run_config.get("num-workers", 0))


def _resolve_taxonomy(context: Context):
    """Resolve the run taxonomy, refusing to guess a scope.

    A silent ``tomato`` fallback would build a ten-class head on a 38-class
    manifest and only surface much later as an out-of-range label error.
    """

    scope = context.run_config.get("taxonomy-scope")
    if not scope:
        raise KeyError(
            "run_config is missing 'taxonomy-scope'; set it in "
            "[tool.flwr.app.config] or pass it from the run launcher"
        )
    return taxonomy_from_scope(str(scope))


def _set_client_seed(msg: Message, context: Context) -> None:
    """Use a stable but distinct augmentation seed for each client and round."""

    base_seed = int(context.run_config["seed"])
    server_round = int(msg.content["config"].get("server-round", 0))
    set_reproducible_seed(base_seed + (_partition_id(context) * 1_000) + server_round)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Load global weights, train only on the client's local images, reply with weights."""

    _set_client_seed(msg, context)
    taxonomy = _resolve_taxonomy(context)
    algorithm = str(context.run_config.get("algorithm", "fedavg")).lower()
    model = build_model(
        str(context.run_config["model-name"]),
        num_classes=len(taxonomy.class_names),
        pretrained=False,
    )
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state)
    restored_bn = _restore_local_batch_norm(context, model, algorithm)
    dataloader = build_dataloader(
        _client_manifest(context, "train_manifest.csv"),
        training=True,
        batch_size=int(context.run_config["batch-size"]),
        num_workers=_num_workers(context),
        dataset_root=_dataset_root(context),
    )
    config = msg.content["config"]
    proximal_mu = float(config.get("proximal-mu", 0.0))

    scaffold_kwargs = _scaffold_train_kwargs(msg, context, model, algorithm)
    moon_kwargs = _moon_train_kwargs(msg, context, model, global_state, algorithm)

    result = train_local(
        model,
        dataloader,
        epochs=int(context.run_config["local-epochs"]),
        learning_rate=float(config["lr"]),
        device=select_device(),
        proximal_mu=proximal_mu,
        **scaffold_kwargs,
        **moon_kwargs,
    )

    metrics: dict[str, object] = {
        "client-id": _partition_id(context),
        "num-examples": result.num_examples,
        "train_loss": result.loss,
    }
    records: dict[str, object] = {"arrays": ArrayRecord(model.state_dict())}

    if algorithm == "fedbn":
        # Evidence that this client trained on its *own* BN statistics rather
        # than the averaged ones. Round 1 legitimately has none stored yet.
        metrics["fedbn_local_bn_tensors"] = restored_bn
        _store_local_batch_norm(context, model)

    if algorithm == "scaffold":
        if result.scaffold_c_i is None:
            raise RuntimeError(
                "SCAFFOLD was selected but local training produced no control "
                "variate; the update would be indistinguishable from FedAvg"
            )
        previous_c_i = _scaffold_previous_c_i(context, model)
        # Report the delta c_i⁺ - c_i, which is what the server accumulates.
        # Torch tensors are used deliberately: ArrayRecord accepts a torch
        # state dict but rejects a dict of raw numpy arrays.
        records[SCAFFOLD_C_DELTA_RECORD] = ArrayRecord(
            {
                name: (value - previous_c_i[name]).detach().cpu()
                for name, value in result.scaffold_c_i.items()
            }
        )
        _store_client_arrays(context, SCAFFOLD_CLIENT_C_RECORD, result.scaffold_c_i)

    if algorithm == "moon":
        if result.moon_contrastive_loss is not None:
            metrics["moon_contrastive_loss"] = result.moon_contrastive_loss
        _store_client_arrays(
            context,
            MOON_PREVIOUS_MODEL_RECORD,
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
        )

    records["metrics"] = MetricRecord(metrics)
    return Message(content=RecordDict(records), reply_to=msg)


def _trainable_names(model) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _store_client_arrays(context: Context, key: str, tensors: dict) -> None:
    """Persist tensors in the client's own state for the next round."""

    context.state[key] = ArrayRecord(tensors)


def _restore_local_batch_norm(context: Context, model, algorithm: str) -> int:
    """FedBN: overwrite the averaged BN tensors with this client's own.

    The server aggregates every parameter, so FedBN has to be enforced here:
    after loading the global weights the client puts its previous local
    batch-norm statistics back, then trains from those.  Returns how many
    tensors were restored — zero in round 1, when nothing is stored yet, and
    the metric the server uses to tell a real FedBN run from plain FedAvg.
    """

    if algorithm != "fedbn":
        return 0
    stored = context.state.array_records.get(FEDBN_LOCAL_BN_RECORD)
    if stored is None:
        return 0
    local_bn = stored.to_torch_state_dict()
    state = model.state_dict()
    expected = batch_norm_parameter_names(state)
    unexpected = set(local_bn) - expected
    if unexpected:
        raise RuntimeError(
            "FedBN client state holds non-batch-norm tensors, first: "
            f"{sorted(unexpected)[0]!r}"
        )
    for name, value in local_bn.items():
        state[name] = value.to(state[name].dtype)
    model.load_state_dict(state)
    return len(local_bn)


def _store_local_batch_norm(context: Context, model) -> None:
    """Keep this client's batch-norm statistics for the next round."""

    state = model.state_dict()
    names = batch_norm_parameter_names(state)
    if not names:
        raise RuntimeError(
            "algorithm is 'fedbn' but the model has no batch-norm layers; "
            "the run would be indistinguishable from FedAvg"
        )
    _store_client_arrays(
        context,
        FEDBN_LOCAL_BN_RECORD,
        {name: state[name].detach().cpu() for name in sorted(names)},
    )


def _scaffold_previous_c_i(context: Context, model) -> dict:
    """Return this client's control variate from the previous round, or zeros."""

    import torch

    stored = context.state.array_records.get(SCAFFOLD_CLIENT_C_RECORD)
    if stored is not None:
        return stored.to_torch_state_dict()
    return {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _scaffold_train_kwargs(
    msg: Message,
    context: Context,
    model,
    algorithm: str,
) -> dict:
    """Build the SCAFFOLD control variates, refusing to degrade into FedAvg."""

    if algorithm != "scaffold":
        return {}
    server_c_record = msg.content.array_records.get(SCAFFOLD_C_RECORD)
    if server_c_record is None:
        raise RuntimeError(
            "algorithm is 'scaffold' but the server sent no control variate "
            f"record {SCAFFOLD_C_RECORD!r}; local training would silently run "
            "as plain FedAvg"
        )
    server_c = server_c_record.to_torch_state_dict()
    client_c = _scaffold_previous_c_i(context, model)
    missing = [name for name in _trainable_names(model) if name not in server_c]
    if missing:
        raise RuntimeError(
            f"SCAFFOLD server control variate is missing {len(missing)} "
            f"parameter(s), first: {missing[0]!r}"
        )
    return {
        "scaffold_control_variate": client_c,
        "scaffold_server_c": server_c,
    }


def _moon_train_kwargs(
    msg: Message,
    context: Context,
    model,
    global_state: dict,
    algorithm: str,
) -> dict:
    """Build the MOON reference models from the global and previous local weights."""

    if algorithm != "moon":
        return {}
    config = msg.content["config"]
    if "moon-mu" not in config or "moon-temperature" not in config:
        raise RuntimeError(
            "algorithm is 'moon' but the server sent no 'moon-mu'/"
            "'moon-temperature'; local training would silently run as FedAvg"
        )
    taxonomy = _resolve_taxonomy(context)
    previous = context.state.array_records.get(MOON_PREVIOUS_MODEL_RECORD)
    if previous is None:
        # Round 1: no previous local model exists yet, so the contrastive term
        # is undefined. This is expected exactly once, and the server only
        # requires the MOON metric from round 2 onwards.
        return {}

    def _clone(state: dict):
        reference = build_model(
            str(context.run_config["model-name"]),
            num_classes=len(taxonomy.class_names),
            pretrained=False,
        )
        reference.load_state_dict(state)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    return {
        "moon_previous_model": _clone(previous.to_torch_state_dict()),
        "moon_global_model": _clone(global_state),
        "moon_temperature": float(config["moon-temperature"]),
        "moon_mu": float(config["moon-mu"]),
    }


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the current global model on the client's held-out validation set."""

    _set_client_seed(msg, context)
    taxonomy = _resolve_taxonomy(context)
    algorithm = str(context.run_config.get("algorithm", "fedavg")).lower()
    model = build_model(
        str(context.run_config["model-name"]),
        num_classes=len(taxonomy.class_names),
        pretrained=False,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Under FedBN the client's real model is global weights + its own BN, so
    # validating the averaged BN would score a model no client ever holds —
    # and that score is what selects the checkpoint.
    _restore_local_batch_norm(context, model, algorithm)
    dataloader = build_dataloader(
        _client_manifest(context, "val_manifest.csv"),
        training=False,
        batch_size=int(context.run_config["batch-size"]),
        num_workers=_num_workers(context),
        dataset_root=_dataset_root(context),
    )
    result = evaluate_model(
        model,
        dataloader,
        device=select_device(),
        class_names=taxonomy.class_names,
        class_groups=taxonomy.class_groups,
    )
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "client-id": _partition_id(context),
                    "num-examples": result.num_examples,
                    **flower_evaluation_values(
                        result,
                        prefix="eval",
                        detailed=False,
                        class_names=taxonomy.class_names,
                    ),
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)
