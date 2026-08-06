"""Single-image inference that executes entirely on the client machine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from cropfed.constants import (
    TOMATO_CLASSES,
    class_group_for_name,
    crop_for_class_name,
)
from cropfed.data.torch_data import build_transforms
from cropfed.ml.checkpoint import load_model_checkpoint
from cropfed.ml.model import build_model
from cropfed.ml.trainer import select_device


def predict_image(
    *,
    checkpoint_path: Path,
    image_path: Path,
    model_name: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Return local top-k predictions; the image is never sent over a network."""

    import torch

    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    inference_started = time.perf_counter()
    loaded = load_model_checkpoint(checkpoint_path)
    class_order = loaded.class_order or TOMATO_CLASSES
    if not 1 <= top_k <= len(class_order):
        raise ValueError(f"top_k must be between 1 and {len(class_order)}")
    resolved_model_name = loaded.model_name or model_name or "mobilenet_v2"
    if model_name is not None and loaded.model_name not in {None, model_name}:
        raise ValueError(
            f"checkpoint model {loaded.model_name!r} does not match "
            f"requested model {model_name!r}"
        )

    device = select_device()
    model = build_model(
        resolved_model_name,
        num_classes=len(class_order),
        pretrained=False,
    )
    model.load_state_dict(loaded.state_dict)
    model.to(device)
    model.eval()

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = build_transforms(training=False)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores, indices = torch.topk(probabilities, k=top_k)
    predictions = [
        {
            "class_id": int(class_id),
            "label": class_order[int(class_id)],
            "group": class_group_for_name(class_order[int(class_id)]),
            "crop": crop_for_class_name(class_order[int(class_id)]),
            "confidence": float(score),
        }
        for score, class_id in zip(
            scores.detach().cpu().tolist(),
            indices.detach().cpu().tolist(),
            strict=True,
        )
    ]
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    primary = predictions[0]
    return {
        "image_name": image_path.name,
        "crop": primary["crop"],
        "predicted_class_id": primary["class_id"],
        "predicted_label": primary["label"],
        "predicted_group": primary["group"],
        "confidence": primary["confidence"],
        "model": resolved_model_name,
        "model_version": loaded.model_version,
        "checkpoint_format_version": loaded.format_version,
        "predictions": predictions,
        "inference_ms": round(inference_ms, 2),
        "warning": (
            "Kết quả chỉ hỗ trợ sàng lọc từ ảnh và không thay thế "
            "chẩn đoán của chuyên gia nông nghiệp."
        ),
        "image_uploaded": False,
    }
