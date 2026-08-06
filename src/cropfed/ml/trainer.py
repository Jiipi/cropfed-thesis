"""Local PyTorch training and evaluation for centralized and FL experiments.

Supports:
- Standard SGD/AdamW (FedAvg)
- FedProx (proximal term)
- SCAFFOLD (control variate correction)
- MOON (model-contrastive loss)
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cropfed.ml.metrics import classification_metrics


@dataclass(frozen=True, slots=True)
class TrainResult:
    loss: float
    num_examples: int
    # SCAFFOLD: the updated client control variate c_i⁺ (not a delta).
    scaffold_c_i: dict[str, Any] | None = None
    # MOON: mean contrastive loss term, evidence the term was actually applied.
    moon_contrastive_loss: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    num_examples: int
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ValidationTrainResult:
    """Training summary after restoring the best validation checkpoint."""

    loss: float
    num_examples: int
    best_epoch: int
    best_validation: EvaluationResult
    history: tuple[dict[str, float | int], ...]


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
    scaffold_control_variate: dict[str, Any] | None = None,
    scaffold_server_c: dict[str, Any] | None = None,
    moon_previous_model: Any | None = None,
    moon_global_model: Any | None = None,
    moon_temperature: float = 0.5,
    moon_mu: float = 1.0,
) -> TrainResult:
    """Train locally with optional FL algorithm-specific regularisation.

    Parameters
    ----------
    proximal_mu:
        FedProx coefficient.  When > 0 the loss is augmented with
        (μ/2) · ‖w - w_global‖².
    scaffold_control_variate / scaffold_server_c:
        SCAFFOLD client/server control variates.  When both are provided
        the gradient is corrected with (c_i - c).
    moon_previous_model / moon_global_model:
        MOON contrastive references.  When both are provided a
        model-contrastive loss term is added to encourage the current
        representation to be closer to the global model than to the
        previous local model.
    moon_temperature:
        MOON temperature τ for the NT-Xent-style contrastive loss.
    moon_mu:
        MOON loss weight μ_moon.
    """

    import torch

    if epochs < 1 or learning_rate <= 0 or proximal_mu < 0:
        raise ValueError("invalid local training hyperparameters")
    use_scaffold = scaffold_control_variate is not None and scaffold_server_c is not None
    use_moon = moon_previous_model is not None and moon_global_model is not None

    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    global_reference = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    # SCAFFOLD initialisation
    if use_scaffold:
        _validate_scaffold_variates(model, scaffold_control_variate, scaffold_server_c)

    # MOON setup
    if use_moon:
        moon_previous_model.to(device)
        moon_previous_model.eval()
        moon_global_model.to(device)
        moon_global_model.eval()
        cos_sim = torch.nn.CosineSimilarity(dim=-1)

    running_loss = 0.0
    batches = 0
    moon_loss_total = 0.0
    moon_batches = 0

    for _ in range(epochs):
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # FedProx proximal term
            if proximal_mu:
                proximal_term = squared_l2_distance_to_reference(
                    model, global_reference
                )
                loss = loss + (proximal_mu / 2.0) * proximal_term

            # MOON model-contrastive loss
            if use_moon and moon_mu > 0:
                moon_loss = _moon_contrastive_loss(
                    model=model,
                    previous_model=moon_previous_model,
                    global_model=moon_global_model,
                    images=images,
                    temperature=moon_temperature,
                    cosine_similarity=cos_sim,
                )
                loss = loss + moon_mu * moon_loss
                moon_loss_total += float(moon_loss.detach().cpu())
                moon_batches += 1

            loss.backward()

            # SCAFFOLD gradient correction
            if use_scaffold:
                _apply_scaffold_correction(
                    model, scaffold_control_variate, scaffold_server_c, learning_rate
                )

            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

    if batches == 0:
        raise ValueError("training dataloader produced no batches")

    # Compute the updated SCAFFOLD client control variate
    scaffold_c_i = None
    if use_scaffold:
        scaffold_c_i = _compute_scaffold_c_i(
            model,
            global_reference,
            scaffold_control_variate,
            scaffold_server_c,
            local_steps=batches,
            learning_rate=learning_rate,
        )

    return TrainResult(
        loss=running_loss / batches,
        num_examples=len(dataloader.dataset),
        scaffold_c_i=scaffold_c_i,
        moon_contrastive_loss=(
            moon_loss_total / moon_batches if moon_batches else None
        ),
    )


def train_with_validation(
    model,
    train_dataloader,
    validation_dataloader,
    *,
    epochs: int,
    learning_rate: float,
    device,
    class_names: Sequence[str],
    class_groups: Sequence[str],
) -> ValidationTrainResult:
    """Train and restore the epoch selected only by validation macro F1.

    The global test set is intentionally absent from this function. Ties in
    macro F1 are resolved by lower validation loss and then the earlier epoch.
    """

    import torch

    if epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid training hyperparameters")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    running_loss = 0.0
    batches = 0
    best_epoch = 0
    best_validation: EvaluationResult | None = None
    best_state = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            running_loss += loss_value
            epoch_batches += 1
            batches += 1
        if epoch_batches == 0:
            raise ValueError("training dataloader produced no batches")

        validation = evaluate_model(
            model,
            validation_dataloader,
            device=device,
            class_names=class_names,
            class_groups=class_groups,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / epoch_batches,
                "validation_loss": validation.loss,
                "validation_accuracy": float(validation.metrics["accuracy"]),
                "validation_macro_f1": float(validation.metrics["macro_f1"]),
            }
        )
        if _validation_is_better(validation, best_validation):
            best_epoch = epoch
            best_validation = validation
            best_state = clone_state_dict(model)

    if batches == 0 or best_validation is None or best_state is None:
        raise ValueError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return ValidationTrainResult(
        loss=running_loss / batches,
        num_examples=len(train_dataloader.dataset),
        best_epoch=best_epoch,
        best_validation=best_validation,
        history=tuple(history),
    )


def _validation_is_better(
    candidate: EvaluationResult,
    incumbent: EvaluationResult | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_f1 = float(candidate.metrics["macro_f1"])
    incumbent_f1 = float(incumbent.metrics["macro_f1"])
    if candidate_f1 != incumbent_f1:
        return candidate_f1 > incumbent_f1
    return candidate.loss < incumbent.loss


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


def evaluate_model(
    model,
    dataloader,
    *,
    device,
    class_names: Sequence[str],
    class_groups: Sequence[str],
) -> EvaluationResult:
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
    resolved_class_names = tuple(class_names)
    resolved_class_groups = tuple(class_groups)
    metrics = classification_metrics(
        targets,
        predictions,
        num_classes=len(resolved_class_names),
        class_names=resolved_class_names,
        healthy_class_ids=tuple(
            index
            for index, group in enumerate(resolved_class_groups)
            if group == "healthy"
        ),
        class_groups=resolved_class_groups,
    )
    return EvaluationResult(
        loss=total_loss / len(targets),
        num_examples=len(targets),
        metrics=metrics,
    )


def clone_state_dict(model):
    """Return a CPU copy suitable for checkpoints or tests."""

    return copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})


# ---------------------------------------------------------------------------
# SCAFFOLD helpers
# ---------------------------------------------------------------------------


def _validate_scaffold_variates(
    model,
    c_i: dict[str, Any],
    c: dict[str, Any],
) -> None:
    """Ensure control variates match the model's trainable parameters."""
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in c_i:
            raise ValueError(f"SCAFFOLD: missing c_i for parameter {name!r}")
        if name not in c:
            raise ValueError(f"SCAFFOLD: missing c for parameter {name!r}")
        if c_i[name].shape != parameter.shape:
            raise ValueError(f"SCAFFOLD: c_i shape mismatch for {name!r}")
        if c[name].shape != parameter.shape:
            raise ValueError(f"SCAFFOLD: c shape mismatch for {name!r}")


def _apply_scaffold_correction(
    model,
    c_i: dict[str, Any],
    c: dict[str, Any],
    learning_rate: float,
) -> None:
    """Add SCAFFOLD correction (c_i - c) to the current gradients.

    This is applied *after* loss.backward() and *before* optimizer.step().

    The SCAFFOLD update rule is:
        w = w - lr * (g + c_i - c)

    Since the optimiser computes w = w - lr * grad, we add (c_i - c)
    directly to grad so that the effective update is:
        w = w - lr * (g + (c_i - c))
    """

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        correction = (c_i[name] - c[name]).to(parameter.device, dtype=parameter.dtype)
        # Add correction directly; the optimiser will multiply by -lr.
        parameter.grad = parameter.grad + correction


def _compute_scaffold_c_i(
    model,
    initial_weights: dict[str, Any],
    c_i: dict[str, Any],
    c: dict[str, Any],
    *,
    local_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Return the updated client control variate ``c_i⁺``.

    SCAFFOLD option II (Karimireddy et al., 2020, Algorithm 1):

        c_i⁺ = c_i - c + (1 / (K · η)) · (w_global - w_local)

    ``K`` is the number of local *optimiser steps* taken this round — not the
    number of examples. Using the example count would scale the correction by
    the batch size and shrink the control variate by that factor.
    """

    steps = max(1, int(local_steps))
    new_c_i: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        w_initial = initial_weights[name].to(parameter.device, dtype=parameter.dtype)
        w_final = parameter.detach()
        ci = c_i[name].to(parameter.device, dtype=parameter.dtype)
        ctrl = c[name].to(parameter.device, dtype=parameter.dtype)
        step = (w_initial - w_final) / (steps * learning_rate)
        new_c_i[name] = (ci - ctrl + step).detach().cpu()
    return new_c_i


# ---------------------------------------------------------------------------
# MOON helpers
# ---------------------------------------------------------------------------


def _moon_contrastive_loss(
    model,
    previous_model,
    global_model,
    images: Any,
    temperature: float,
    cosine_similarity,
) -> Any:
    """Compute MOON model-contrastive loss for a batch.

    Loss = -log(
        exp(sim(z, z_glob) / τ) /
        (exp(sim(z, z_glob) / τ) + exp(sim(z, z_prev) / τ))
    )

    where z is the representation (pre-classifier) of the current model,
    z_glob is from the global model, and z_prev is from the previous local
    model.

    Uses the penultimate layer (before classifier) as the representation.
    For MobileNetV2 this is the output of the adaptive avg pool.
    """
    import torch
    import torch.nn.functional as F

    z = _extract_representation(model, images)
    with torch.no_grad():
        z_glob = _extract_representation(global_model, images)
        z_prev = _extract_representation(previous_model, images)

    z = F.normalize(z, dim=1)
    z_glob = F.normalize(z_glob, dim=1)
    z_prev = F.normalize(z_prev, dim=1)

    sim_glob = cosine_similarity(z, z_glob) / temperature
    sim_prev = cosine_similarity(z, z_prev) / temperature

    # Concatenate [sim_glob, sim_prev] along a new dimension
    logits = torch.stack([sim_glob, sim_prev], dim=1)
    # Label 0 = "global is the positive pair"
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def _extract_representation(model, images: Any) -> Any:
    """Extract the feature representation before the final classifier layer.

    The returned tensor stays attached to the autograd graph: MOON needs a
    gradient through the *current* model's representation. Callers wrap the
    reference models in ``torch.no_grad()`` themselves.

    For MobileNetV2/V3 and EfficientNet: ``features`` + global average pool.
    For ResNet18: the input to ``fc``, captured with a forward hook.
    """
    import torch

    # --- MobileNetV2 / MobileNetV3-Small / EfficientNet-B0 ---
    if hasattr(model, "features"):
        features = model.features(images)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(pooled, 1)

    # --- ResNet-style ---
    if hasattr(model, "fc"):
        captured: list[Any] = []

        def _hook(_module, _input, _output) -> None:
            # _input is a tuple of one tensor: the features feeding ``fc``.
            # It must NOT be detached — MOON backpropagates through it.
            captured.append(_input[0])

        handle = model.fc.register_forward_hook(_hook)
        try:
            _ = model(images)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(
                "MOON could not capture a representation from the model's "
                "final layer; the contrastive loss would be undefined"
            )
        return torch.flatten(captured[0], 1)

    raise TypeError(
        "MOON requires a backbone exposing either 'features' or 'fc'; "
        f"{type(model).__name__} exposes neither"
    )
