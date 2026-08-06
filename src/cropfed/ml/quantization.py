"""Model quantisation for edge-device deployment.

Supports post-training quantisation (PTQ) and quantisation-aware training (QAT)
for MobileNetV2, MobileNetV3-Small, and EfficientNet-B0 backbones.

The quantised model is suitable for deployment on:
- CPU-only edge devices (ARM, x86)
- Mobile devices (via ONNX Runtime, TensorFlow Lite, or Core ML)
- Low-power agricultural sensors

References
----------
- Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference", CVPR 2018
- PyTorch quantisation docs: https://pytorch.org/docs/stable/quantization.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QuantisationResult:
    """Metrics collected after quantising a model."""

    original_size_bytes: int
    quantised_size_bytes: int
    compression_ratio: float
    original_parameters: int
    quantised_parameters: int


def quantise_model_static(
    model,
    calibration_dataloader,
    *,
    device,
    dtype: str = "qint8",
) -> Any:
    """Apply static post-training quantisation (PTQ).

    The model is calibrated using a few batches from *calibration_dataloader*
    and then converted to INT8.  This is the simplest path to smaller models
    and faster CPU inference.

    Parameters
    ----------
    model:
        A trained float32 PyTorch model.
    calibration_dataloader:
        DataLoader with a representative sample of training/validation data
        (typically 100-500 examples).
    device:
        CPU device (quantisation in PyTorch is CPU-only).
    dtype:
        Target dtype — 'qint8' for int8 quantisation.

    Returns
    -------
    torch.nn.Module
        The quantised model ready for inference.
    """
    import torch
    import torch.quantization as quant

    if dtype != "qint8":
        raise ValueError("only 'qint8' static quantisation is supported")

    model.to("cpu")
    model.eval()

    # Fuse common patterns (Conv + BN + ReLU) for better quantisation accuracy
    model = _fuse_modules(model)

    # Configure quantisation
    model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(model, inplace=True)

    # Calibrate with representative data
    with torch.inference_mode():
        for images, _ in calibration_dataloader:
            model(images.to("cpu"))

    quant.convert(model, inplace=True)
    return model


def quantise_model_dynamic(model, dtype: str = "qint8") -> Any:
    """Apply dynamic quantisation — no calibration data needed.

    Only Linear layers are quantised; weights are converted to INT8 and
    activations are quantised dynamically at inference time.  This is
    faster than PTQ to apply and works well when a calibration dataset
    is not available.

    Parameters
    ----------
    model:
        A trained float32 PyTorch model.
    dtype:
        Target dtype for weights.

    Returns
    -------
    torch.nn.Module
        The dynamically quantised model.
    """
    import torch

    model.to("cpu")
    model.eval()

    dtype_map = {
        "qint8": torch.qint8,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"dtype must be one of {list(dtype_map)}")

    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch_dtype,
    )


def quantisation_aware_training(
    model,
    train_dataloader,
    validation_dataloader,
    *,
    epochs: int,
    learning_rate: float,
    device,
) -> Any:
    """Run quantisation-aware training (QAT) for better accuracy than PTQ.

    QAT simulates quantisation during training so the model learns to
    compensate for quantisation error.  This typically yields 1-3% better
    accuracy than static PTQ.

    Parameters
    ----------
    model:
        A pre-trained float32 model.
    train_dataloader:
        Training data loader.
    validation_dataloader:
        Validation data loader.
    epochs:
        Number of QAT epochs (typically 5-10).
    learning_rate:
        Learning rate (typically 1e-4 to 1e-5, much lower than normal training).
    device:
        CPU device (QAT requires CPU in PyTorch).

    Returns
    -------
    tuple[torch.nn.Module, dict]
        (quantised_model, training_history)
    """
    import torch
    import torch.quantization as quant

    model.to("cpu")
    model.train()

    model = _fuse_modules(model)
    model.qconfig = quant.get_default_qat_qconfig("fbgemm")
    quant.prepare_qat(model, inplace=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for images, labels in train_dataloader:
            images = images.to("cpu")
            labels = labels.to("cpu")
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.inference_mode():
            for images, labels in validation_dataloader:
                val_loss += float(criterion(model(images.to("cpu")), labels.to("cpu")))
                val_batches += 1

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, batches),
            "val_loss": val_loss / max(1, val_batches),
        })

    model.to("cpu")
    model.eval()
    quant.convert(model, inplace=True)
    return model, history


def measure_model_size(model, state_dict: dict[str, Any] | None = None) -> int:
    """Return the serialised model size in bytes (parameters only)."""
    import io
    import torch

    buffer = io.BytesIO()
    if state_dict is not None:
        torch.save(state_dict, buffer)
    else:
        torch.save(model.state_dict(), buffer)
    return buffer.tell()


def compare_quantisation(
    float_model,
    quantised_model,
    calibration_dataloader,
    *,
    device,
) -> QuantisationResult:
    """Measure size reduction from quantisation."""

    float_state = {k: v.cpu() for k, v in float_model.state_dict().items()}
    quant_state = {k: v.cpu() for k, v in quantised_model.state_dict().items()}

    original_bytes = measure_model_size(float_model, float_state)
    quantised_bytes = measure_model_size(quantised_model, quant_state)

    original_params = sum(p.numel() for p in float_model.parameters())
    quantised_params = sum(p.numel() for p in quantised_model.parameters())

    return QuantisationResult(
        original_size_bytes=original_bytes,
        quantised_size_bytes=quantised_bytes,
        compression_ratio=original_bytes / max(1, quantised_bytes),
        original_parameters=original_params,
        quantised_parameters=quantised_params,
    )


def _fuse_modules(model):
    """Fuse Conv+BN+ReLU sequences for better quantisation.

    This is a best-effort operation — if the model doesn't have the
    expected module structure, fusion is skipped silently.
    """
    import torch

    # MobileNetV2 uses ConvBNReLU blocks in its 'features' sequential
    if hasattr(model, "features"):
        try:
            model = torch.quantization.fuse_modules(model, [
                ["features.0.0", "features.0.1", "features.0.2"],
            ])
        except (AttributeError, TypeError, AssertionError):
            pass
        # Try to fuse remaining Conv+BN pairs in the features
        for i in range(1, 19):  # MobileNetV2 has 18 layers
            try:
                model = torch.quantization.fuse_modules(model, [
                    [f"features.{i}.conv.0.0", f"features.{i}.conv.0.1"],
                    [f"features.{i}.conv.1.0", f"features.{i}.conv.1.1"],
                    [f"features.{i}.conv.2", f"features.{i}.conv.3"],
                ])
            except (AttributeError, TypeError, AssertionError, IndexError):
                pass

    return model