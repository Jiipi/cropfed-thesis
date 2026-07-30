"""Local PyTorch training and evaluation for centralized and FL experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from cropfed.constants import TOMATO_CLASS_GROUPS, TOMATO_CLASSES
from cropfed.ml.metrics import classification_metrics


@dataclass(frozen=True, slots=True)
class TrainResult:
    loss: float
    num_examples: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    num_examples: int
    metrics: dict[str, Any]


def select_device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_reproducible_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_local(
    model,
    dataloader,
    *,
    epochs: int,
    learning_rate: float,
    device,
    proximal_mu: float = 0.0,
) -> TrainResult:
    """Train locally; add the FedProx term when ``proximal_mu`` is positive."""

    import torch

    if epochs < 1 or learning_rate <= 0 or proximal_mu < 0:
        raise ValueError("invalid local training hyperparameters")
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    global_reference = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    running_loss = 0.0
    batches = 0

    for _ in range(epochs):
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if proximal_mu:
                proximal_term = squared_l2_distance_to_reference(
                    model, global_reference
                )
                loss = loss + (proximal_mu / 2.0) * proximal_term
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

    if batches == 0:
        raise ValueError("training dataloader produced no batches")
    return TrainResult(
        loss=running_loss / batches,
        num_examples=len(dataloader.dataset),
    )


def squared_l2_distance_to_reference(model, reference_parameters):
    """Return the squared L2 distance used by the FedProx local objective."""

    import torch

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise ValueError("model has no trainable parameters")
    if set(trainable) != set(reference_parameters):
        raise ValueError("reference parameters do not match trainable model parameters")

    first_parameter = next(iter(trainable.values()))
    distance = torch.zeros(
        (),
        device=first_parameter.device,
        dtype=first_parameter.dtype,
    )
    for name, parameter in trainable.items():
        reference = reference_parameters[name]
        if parameter.shape != reference.shape:
            raise ValueError(f"reference shape mismatch for parameter {name!r}")
        distance = distance + torch.sum((parameter - reference) ** 2)
    return distance


def evaluate_model(model, dataloader, *, device) -> EvaluationResult:
    import torch

    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(criterion(logits, labels).detach().cpu())
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

    if not targets:
        raise ValueError("evaluation dataloader produced no examples")
    metrics = classification_metrics(
        targets,
        predictions,
        num_classes=len(TOMATO_CLASSES),
        class_names=TOMATO_CLASSES,
        healthy_class_id=0,
        class_groups=TOMATO_CLASS_GROUPS,
    )
    return EvaluationResult(
        loss=total_loss / len(targets),
        num_examples=len(targets),
        metrics=metrics,
    )


def clone_state_dict(model):
    """Return a CPU copy suitable for checkpoints or tests."""

    return copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
