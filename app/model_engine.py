from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

from .vision_models import (
    available_model_options,
    build_model,
    get_imagenet_categories,
    get_model_spec,
    get_inference_transform,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "experiments" / "checkpoints"
ANNOTATED_DIR = DATA_DIR / "annotated"
STRESS_DIR = DATA_DIR / "stress"

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
STRESS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_MODE = "trained-road7-checkpoint"
FALLBACK_MODE = "torchvision-imagenet-fallback"
HYBRID_MODE = "hybrid-road7-checkpoint+imagenet"
PRIOR_OVERRIDE_THRESHOLD = 0.55

CLASS_META = {
    "person": {"label": "行人", "display": "Person", "color": "#ef4444"},
    "bicycle": {"label": "自行车", "display": "Bicycle", "color": "#22c55e"},
    "car": {"label": "汽车", "display": "Car", "color": "#3b82f6"},
    "motorcycle": {"label": "摩托车", "display": "Motorcycle", "color": "#f59e0b"},
    "bus": {"label": "公交车", "display": "Bus", "color": "#8b5cf6"},
    "truck": {"label": "卡车", "display": "Truck", "color": "#06b6d4"},
    "traffic_light": {"label": "交通信号灯", "display": "Traffic Light", "color": "#ec4899"},
    "unknown": {"label": "未识别", "display": "Unknown", "color": "#64748b"},
}

CLASS_ALIASES = {
    "pedestrian": "person",
    "traffic_sign_light": "traffic_light",
    "traffic light": "traffic_light",
}

IMAGENET_TO_CLASS = {
    "sports car": "car",
    "racer": "car",
    "convertible": "car",
    "cab": "car",
    "jeep": "car",
    "limousine": "car",
    "minivan": "car",
    "station wagon": "car",
    "car wheel": "car",
    "grille": "car",
    "pickup": "truck",
    "moving van": "truck",
    "tow truck": "truck",
    "trailer truck": "truck",
    "fire engine": "truck",
    "garbage truck": "truck",
    "school bus": "bus",
    "trolleybus": "bus",
    "minibus": "bus",
    "motor scooter": "motorcycle",
    "moped": "motorcycle",
    "mountain bike": "bicycle",
    "bicycle-built-for-two": "bicycle",
    "unicycle": "bicycle",
    "traffic light": "traffic_light",
    "street sign": "traffic_light",
    "parking meter": "traffic_light",
    "pedestrian crossing": "traffic_light",
    "person": "person",
    "groom": "person",
    "scuba diver": "person",
    "ballplayer": "person",
}

STRESS_SCENARIOS = {
    "low_light": {
        "label": "低照度",
        "description": "压低亮度并叠加轻微噪声，模拟夜间和阴天场景。",
    },
    "gaussian_noise": {
        "label": "高斯噪声",
        "description": "叠加明显噪声，模拟传感器噪声和压缩失真。",
    },
    "partial_crop": {
        "label": "局部裁剪",
        "description": "裁掉边缘区域，模拟目标被遮挡或拍摄不完整。",
    },
}

_lock = threading.Lock()
_model_cache: dict[str, torch.nn.Module] = {}
_transform_cache: dict[str, Any] = {}
_label_cache: dict[str, list[str]] = {}
_mode_cache: dict[str, str] = {}
_imagenet_model_cache: dict[str, torch.nn.Module] = {}
_imagenet_transform_cache: dict[str, Any] = {}
_imagenet_label_cache: dict[str, list[str]] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class InferenceResult:
    predicted_class: str
    predicted_label: str
    confidence: float
    model_mode: str
    detections: list[dict[str, Any]]
    annotated_path: Optional[str]
    gradcam_enabled: bool
    model_name: str
    model_display_name: str


class EngineUnavailable(RuntimeError):
    pass


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _seed_from_text(text: str) -> int:
    return sum(ord(ch) for ch in text) % (2**32)


def _canonical_class_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = CLASS_ALIASES.get(normalized, normalized)
    return normalized if normalized in CLASS_META else "unknown"


def _checkpoint_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"{model_name}.pt"


def get_available_models() -> list[dict[str, str]]:
    items = []
    for item in available_model_options():
        source = "trained-checkpoint" if _checkpoint_path(item["name"]).exists() else "imagenet-fallback"
        items.append({**item, "source": source})
    return items


def _load_model_bundle(model_name: str) -> tuple[torch.nn.Module, Any, list[str], str]:
    if model_name not in _model_cache:
        with _lock:
            if model_name not in _model_cache:
                checkpoint_path = _checkpoint_path(model_name)
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    class_names = [_canonical_class_name(name) for name in checkpoint.get("class_names", [])]
                    if not class_names:
                        raise EngineUnavailable(f"{model_name} 的 checkpoint 缺少类别信息")
                    model, _ = build_model(
                        model_name,
                        num_classes=len(class_names),
                        pretrained=True,
                        freeze_backbone=False,
                    )
                    model.load_state_dict(checkpoint["state_dict"])
                    mode = CHECKPOINT_MODE
                    labels = class_names
                else:
                    model, _ = build_model(model_name, pretrained=True)
                    labels = list(get_imagenet_categories(model_name))
                    mode = FALLBACK_MODE

                model.eval()
                model.to(_device)
                _model_cache[model_name] = model
                _transform_cache[model_name] = get_inference_transform(model_name)
                _label_cache[model_name] = labels
                _mode_cache[model_name] = mode

    return (
        _model_cache[model_name],
        _transform_cache[model_name],
        _label_cache[model_name],
        _mode_cache[model_name],
    )


def _load_imagenet_bundle(model_name: str) -> tuple[torch.nn.Module, Any, list[str]]:
    if model_name not in _imagenet_model_cache:
        with _lock:
            if model_name not in _imagenet_model_cache:
                model, _ = build_model(model_name, pretrained=True)
                model.eval()
                model.to(_device)
                _imagenet_model_cache[model_name] = model
                _imagenet_transform_cache[model_name] = get_inference_transform(model_name)
                _imagenet_label_cache[model_name] = list(get_imagenet_categories(model_name))

    return (
        _imagenet_model_cache[model_name],
        _imagenet_transform_cache[model_name],
        _imagenet_label_cache[model_name],
    )


def _find_last_conv(module: torch.nn.Module) -> torch.nn.Module:
    last_conv: Optional[torch.nn.Module] = None
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise EngineUnavailable("未找到可用于 Grad-CAM 的卷积层")
    return last_conv


def _generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    original_image: Image.Image,
) -> Image.Image:
    target_layer = _find_last_conv(model)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output.detach())

    def backward_hook(
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        gradients.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        logits[:, class_index].sum().backward()

        if not activations or not gradients:
            raise EngineUnavailable("Grad-CAM 未捕获到有效特征")

        activation = activations[-1][0]
        gradient = gradients[-1][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activation).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()

        heat = Image.fromarray(np.uint8(cam * 255), mode="L").resize(original_image.size, Image.BILINEAR)
        heat_array = np.asarray(heat).astype("float32") / 255.0
        base = np.asarray(original_image.convert("RGB")).astype("float32")

        overlay = np.zeros_like(base)
        overlay[..., 0] = 255 * heat_array
        overlay[..., 1] = 96 * (1.0 - heat_array)
        overlay[..., 2] = 32 * (1.0 - heat_array)

        mixed = np.clip(base * 0.62 + overlay * 0.38, 0, 255).astype("uint8")
        return Image.fromarray(mixed)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _build_detection(class_name: str, score: float, raw_label: str) -> dict[str, Any]:
    meta = CLASS_META.get(class_name, CLASS_META["unknown"])
    return {
        "class_name": class_name,
        "label": meta["display"],
        "display_label": meta["label"],
        "score": score,
        "box": [0, 0, 0, 0],
        "raw_label": raw_label,
    }


def _run_finetuned_inference(labels: list[str], probs: torch.Tensor) -> tuple[list[dict[str, Any]], str, float, int]:
    top_k = min(5, len(labels))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    detections: list[dict[str, Any]] = []

    chosen_index = int(top_indices[0].item())
    chosen_class = _canonical_class_name(labels[chosen_index])
    chosen_score = float(top_probs[0].item())

    for prob, idx_tensor in zip(top_probs, top_indices):
        idx = int(idx_tensor.item())
        raw_label = labels[idx] if idx < len(labels) else f"class-{idx}"
        class_name = _canonical_class_name(raw_label)
        detections.append(_build_detection(class_name, float(prob.item()), raw_label))

    return detections, chosen_class, chosen_score, chosen_index


def _road_class_order() -> list[str]:
    return [name for name in CLASS_META.keys() if name != "unknown"]


def _project_probs_to_road_classes(labels: list[str], probs: torch.Tensor, *, use_imagenet_mapping: bool) -> torch.Tensor:
    road_labels = _road_class_order()
    projected = torch.zeros(len(road_labels), dtype=probs.dtype, device=probs.device)
    road_index = {name: idx for idx, name in enumerate(road_labels)}

    for idx, score in enumerate(probs):
        raw_label = labels[idx] if idx < len(labels) else f"class-{idx}"
        if use_imagenet_mapping:
            class_name = IMAGENET_TO_CLASS.get(raw_label.lower().strip())
        else:
            class_name = _canonical_class_name(raw_label)
            if class_name == "unknown":
                class_name = None
        if class_name in road_index:
            projected[road_index[class_name]] += score

    total = projected.sum()
    if float(total.item()) > 0:
        projected = projected / total
    return projected


def _build_road_detections(probs: torch.Tensor) -> tuple[list[dict[str, Any]], str, float]:
    road_labels = _road_class_order()
    top_k = min(5, len(road_labels))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    detections: list[dict[str, Any]] = []

    chosen_class = road_labels[int(top_indices[0].item())]
    chosen_score = float(top_probs[0].item())

    for prob, idx_tensor in zip(top_probs, top_indices):
        class_name = road_labels[int(idx_tensor.item())]
        detections.append(_build_detection(class_name, float(prob.item()), class_name))

    return detections, chosen_class, chosen_score


def _run_fallback_inference(labels: list[str], probs: torch.Tensor) -> tuple[list[dict[str, Any]], str, float, int]:
    top_probs, top_indices = torch.topk(probs, k=5)
    detections: list[dict[str, Any]] = []

    chosen_class = "unknown"
    chosen_score = float(top_probs[0].item())
    chosen_index = int(top_indices[0].item())

    for prob, idx_tensor in zip(top_probs, top_indices):
        idx = int(idx_tensor.item())
        raw_label = labels[idx] if idx < len(labels) else f"class-{idx}"
        mapped = IMAGENET_TO_CLASS.get(raw_label.lower().strip(), "unknown")
        score = float(prob.item())
        detections.append(_build_detection(mapped, score, raw_label))
        if mapped != "unknown" and chosen_class == "unknown":
            chosen_class = mapped
            chosen_score = score
            chosen_index = idx

    return detections, chosen_class, chosen_score, chosen_index


def run_inference(
    image_path: Union[str, Path],
    model_name: str = "mobilenet_v2",
    with_gradcam: bool = True,
) -> InferenceResult:
    path = Path(image_path)
    try:
        model, transform, labels, mode = _load_model_bundle(model_name)
        spec = get_model_spec(model_name)
        original = Image.open(path).convert("RGB")
        input_tensor = transform(original).unsqueeze(0).to(_device)

        with torch.inference_mode():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        if mode == CHECKPOINT_MODE:
            detections, chosen_class, chosen_score, chosen_index = _run_finetuned_inference(labels, probs)
            projected_checkpoint = _project_probs_to_road_classes(labels, probs, use_imagenet_mapping=False)

            prior_model, prior_transform, prior_labels = _load_imagenet_bundle(model_name)
            prior_tensor = prior_transform(original).unsqueeze(0).to(_device)
            with torch.inference_mode():
                prior_logits = prior_model(prior_tensor)
                prior_probs = torch.softmax(prior_logits, dim=1)[0]
            projected_prior = _project_probs_to_road_classes(prior_labels, prior_probs, use_imagenet_mapping=True)
            prior_detections, prior_class, prior_score = _build_road_detections(projected_prior)

            if prior_class != chosen_class and prior_score >= PRIOR_OVERRIDE_THRESHOLD:
                detections = prior_detections
                chosen_class = prior_class
                chosen_score = prior_score
                if chosen_class in labels:
                    chosen_index = labels.index(chosen_class)
                else:
                    chosen_index = int(torch.argmax(projected_checkpoint).item())
                mode = HYBRID_MODE
        else:
            detections, chosen_class, chosen_score, chosen_index = _run_fallback_inference(labels, probs)

        annotated_path: Optional[str] = None
        if with_gradcam:
            cam_image = _generate_gradcam(
                model,
                input_tensor.clone().requires_grad_(True),
                chosen_index,
                original,
            )
            target_name = f"{path.stem}_{_safe_name(model_name)}_gradcam.jpg"
            output_path = ANNOTATED_DIR / target_name
            cam_image.save(output_path, quality=92)
            annotated_path = str(output_path.relative_to(ROOT))

        predicted_class = chosen_class if chosen_class in CLASS_META else "unknown"
        return InferenceResult(
            predicted_class=predicted_class,
            predicted_label=CLASS_META[predicted_class]["label"],
            confidence=chosen_score,
            model_mode=f"{mode}:{model_name}",
            detections=detections,
            annotated_path=annotated_path,
            gradcam_enabled=with_gradcam,
            model_name=model_name,
            model_display_name=spec.display_name,
        )
    except Exception as exc:
        raise EngineUnavailable(f"推理引擎初始化或执行失败：{exc}") from exc


def _add_gaussian_noise(image: Image.Image, strength: float, seed_text: str) -> Image.Image:
    array = np.asarray(image).astype("float32") / 255.0
    rng = np.random.default_rng(_seed_from_text(seed_text))
    noise = rng.normal(0, strength, array.shape)
    array = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((array * 255).astype("uint8"))


def _apply_stress_scenario(image: Image.Image, scenario_name: str, seed_text: str) -> Image.Image:
    result = image.convert("RGB")

    if scenario_name == "low_light":
        result = ImageEnhance.Brightness(result).enhance(0.38)
        result = ImageEnhance.Contrast(result).enhance(0.92)
        result = _add_gaussian_noise(result, 0.03, seed_text)
        return result

    if scenario_name == "gaussian_noise":
        return _add_gaussian_noise(result, 0.09, seed_text)

    if scenario_name == "partial_crop":
        width, height = result.size
        left = int(width * 0.08)
        top = int(height * 0.06)
        right = int(width * 0.9)
        bottom = int(height * 0.84)
        result = result.crop((left, top, right, bottom))
        result = result.resize((width, height), Image.BILINEAR)
        result = result.filter(ImageFilter.GaussianBlur(radius=0.6))
        return result

    raise EngineUnavailable(f"不支持的压力测试场景：{scenario_name}")


def run_stress_test(
    image_path: Union[str, Path],
    model_name: str = "mobilenet_v2",
) -> list[dict[str, Any]]:
    source_path = Path(image_path)
    original = Image.open(source_path).convert("RGB")
    items: list[dict[str, Any]] = []

    for scenario_name, meta in STRESS_SCENARIOS.items():
        seed_text = f"{source_path.stem}:{scenario_name}"
        scenario_image = _apply_stress_scenario(original.copy(), scenario_name, seed_text)
        output_name = f"{source_path.stem}_{scenario_name}_{_seed_from_text(seed_text):08x}.jpg"
        output_path = STRESS_DIR / output_name
        scenario_image.save(output_path, quality=92)

        result = run_inference(output_path, model_name=model_name, with_gradcam=False)
        items.append(
            {
                "scene": scenario_name,
                "scene_label": meta["label"],
                "scene_description": meta["description"],
                "image_path": str(output_path.relative_to(ROOT)),
                "predicted_class": result.predicted_class,
                "predicted_label": result.predicted_label,
                "confidence": round(result.confidence, 4),
                "model_mode": result.model_mode,
                "model_name": result.model_name,
                "model_display_name": result.model_display_name,
                "detections": result.detections,
            }
        )

    return items
