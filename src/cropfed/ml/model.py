"""Image-classification model factory."""

from __future__ import annotations


def build_model(
    model_name: str = "mobilenet_v2",
    *,
    num_classes: int = 10,
    pretrained: bool = True,
):
    """Build the primary MobileNetV2 or the ResNet18 comparison model."""

    from torch import nn
    from torchvision.models import (
        MobileNet_V2_Weights,
        ResNet18_Weights,
        mobilenet_v2,
        resnet18,
    )

    if model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        input_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(input_features, num_classes)
        return model
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes)
        return model
    raise ValueError("model_name must be 'mobilenet_v2' or 'resnet18'")


def count_trainable_parameters(model) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
