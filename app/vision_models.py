from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v2,
    resnet50,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    builder: Callable[..., nn.Module]
    weights: object
    feature_dim: int
    family: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet50": ModelSpec(
        name="resnet50",
        display_name="ResNet50",
        builder=resnet50,
        weights=ResNet50_Weights.DEFAULT,
        feature_dim=2048,
        family="resnet",
    ),
    "mobilenet_v2": ModelSpec(
        name="mobilenet_v2",
        display_name="MobileNetV2",
        builder=mobilenet_v2,
        weights=MobileNet_V2_Weights.DEFAULT,
        feature_dim=1280,
        family="mobilenet",
    ),
    "efficientnet_b0": ModelSpec(
        name="efficientnet_b0",
        display_name="EfficientNet-B0",
        builder=efficientnet_b0,
        weights=EfficientNet_B0_Weights.DEFAULT,
        feature_dim=1280,
        family="efficientnet",
    ),
}


def available_model_options() -> list[dict[str, str]]:
    return [
        {"name": spec.name, "display_name": spec.display_name}
        for spec in MODEL_SPECS.values()
    ]


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"不支持的模型：{model_name}")
    return MODEL_SPECS[model_name]


def build_model(
    model_name: str,
    *,
    num_classes: Optional[int] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> tuple[nn.Module, object]:
    spec = get_model_spec(model_name)
    weights = spec.weights if pretrained else None
    model = spec.builder(weights=weights)

    if num_classes is not None:
        if model_name == "resnet50":
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "mobilenet_v2":
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == "efficientnet_b0":
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"未实现分类头替换：{model_name}")

    if freeze_backbone and num_classes is not None:
        for param in model.parameters():
            param.requires_grad = False
        if model_name == "resnet50":
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model, weights


def get_inference_transform(model_name: str):
    spec = get_model_spec(model_name)
    return spec.weights.transforms()


def get_imagenet_categories(model_name: str) -> list[str]:
    spec = get_model_spec(model_name)
    return list(spec.weights.meta.get("categories", []))


def model_parameter_stats(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


class AddGaussianNoise(nn.Module):
    def __init__(self, std: float = 0.03) -> None:
        super().__init__()
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"
