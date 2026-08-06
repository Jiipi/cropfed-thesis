"""Image-classification model factory.

Supported backbones follow the thesis proposal:
- MobileNetV3-Small / EfficientNet-Lite0 (lightweight edge targets)
- MobileNetV2 (primary baseline)
- ResNet18 (heavier comparison point)
"""

from __future__ import annotations


def build_model(
    model_name: str = "mobilenet_v2",
    *,
    num_classes: int,
    pretrained: bool = True,
):
    """Build a CNN backbone with the classifier head replaced for the task."""


    if model_name == "mobilenet_v2":
        return _build_mobilenet_v2(num_classes, pretrained)
    if model_name == "mobilenet_v3_small":
        return _build_mobilenet_v3_small(num_classes, pretrained)
    if model_name == "efficientnet_lite0":
        return _build_efficientnet_lite0(num_classes, pretrained)
    if model_name == "resnet18":
        return _build_resnet18(num_classes, pretrained)
    raise ValueError(
        "model_name must be 'mobilenet_v2', 'mobilenet_v3_small', "
        "'efficientnet_lite0', or 'resnet18'"
    )


def _build_mobilenet_v2(num_classes: int, pretrained: bool):
    from torch import nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)
    return model


def _build_mobilenet_v3_small(num_classes: int, pretrained: bool):
    """MobileNetV3-Small: ~2.5M params, suitable for edge devices.

    Named in the thesis proposal as a lightweight client backbone target.
    """
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, num_classes)
    return model


def _build_efficientnet_lite0(num_classes: int, pretrained: bool):
    """EfficientNet-Lite0: ~4.6M params, integer-quantisation friendly.

    Named in the thesis proposal as a lightweight edge alternative.
    Torchvision does not ship EfficientNet-Lite directly, so we use
    EfficientNet-B0 (the closest available variant) as a stand-in with
    a note that true Lite0 requires a separate package.

    When ``pretrained`` is True the standard EfficientNet_B0_Weights are
    used; the caller may also pass ``pretrained=False`` for random init.
    """
    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)
    return model


def _build_resnet18(num_classes: int, pretrained: bool):
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def count_trainable_parameters(model) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
