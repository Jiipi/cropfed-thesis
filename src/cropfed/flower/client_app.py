"""Agricultural client: local image access, training and evaluation only."""

from __future__ import annotations

from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from cropfed.constants import TOMATO_CLASSES
from cropfed.data.torch_data import build_dataloader
from cropfed.ml.model import build_model
from cropfed.ml.reporting import flower_evaluation_values
from cropfed.ml.trainer import (
    evaluate_model,
    select_device,
    set_reproducible_seed,
    train_local,
)

app = ClientApp()


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


def _set_client_seed(msg: Message, context: Context) -> None:
    """Use a stable but distinct augmentation seed for each client and round."""

    base_seed = int(context.run_config["seed"])
    server_round = int(msg.content["config"].get("server-round", 0))
    set_reproducible_seed(base_seed + (_partition_id(context) * 1_000) + server_round)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Load global weights, train only on the client's local images, reply with weights."""

    _set_client_seed(msg, context)
    model = build_model(
        str(context.run_config["model-name"]),
        num_classes=len(TOMATO_CLASSES),
        pretrained=False,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    dataloader = build_dataloader(
        _client_manifest(context, "train_manifest.csv"),
        training=True,
        batch_size=int(context.run_config["batch-size"]),
    )
    proximal_mu = float(msg.content["config"].get("proximal-mu", 0.0))
    result = train_local(
        model,
        dataloader,
        epochs=int(context.run_config["local-epochs"]),
        learning_rate=float(msg.content["config"]["lr"]),
        device=select_device(),
        proximal_mu=proximal_mu,
    )
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "client-id": _partition_id(context),
                    "num-examples": result.num_examples,
                    "train_loss": result.loss,
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the current global model on the client's held-out validation set."""

    _set_client_seed(msg, context)
    model = build_model(
        str(context.run_config["model-name"]),
        num_classes=len(TOMATO_CLASSES),
        pretrained=False,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    dataloader = build_dataloader(
        _client_manifest(context, "val_manifest.csv"),
        training=False,
        batch_size=int(context.run_config["batch-size"]),
    )
    result = evaluate_model(model, dataloader, device=select_device())
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
                    ),
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)
