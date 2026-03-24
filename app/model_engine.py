from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mobilenetv2-7.onnx"
LABELS_PATH = MODEL_DIR / "imagenet-simple-labels.json"
MODEL_URL = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

CLASS_META = {
    "pedestrian": {"label": "行人", "display": "Pedestrian", "color": "#ef4444"},
    "bicycle": {"label": "自行车", "display": "Bicycle", "color": "#22c55e"},
    "car": {"label": "汽车", "display": "Car", "color": "#3b82f6"},
    "motorcycle": {"label": "摩托车", "display": "Motorcycle", "color": "#f59e0b"},
    "bus": {"label": "公交车", "display": "Bus", "color": "#8b5cf6"},
    "truck": {"label": "卡车", "display": "Truck", "color": "#06b6d4"},
    "traffic_sign_light": {"label": "交通标志/信号灯", "display": "Traffic Sign/Light", "color": "#ec4899"},
    "unknown": {"label": "未识别", "display": "Unknown", "color": "#64748b"},
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
    "traffic light": "traffic_sign_light",
    "street sign": "traffic_sign_light",
    "parking meter": "traffic_sign_light",
    "pedestrian crossing": "traffic_sign_light",
    "groom": "pedestrian",
    "scuba diver": "pedestrian",
    "ballplayer": "pedestrian",
}

_lock = threading.Lock()
_session: ort.InferenceSession | None = None
_labels: list[str] | None = None


@dataclass
class InferenceResult:
    predicted_class: str
    predicted_label: str
    confidence: float
    model_mode: str
    detections: list[dict[str, Any]]
    annotated_path: str | None


class EngineUnavailable(RuntimeError):
    pass


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def _load_resources() -> tuple[ort.InferenceSession, list[str]]:
    global _session, _labels
    if _session is None or _labels is None:
        with _lock:
            if _session is None or _labels is None:
                _download_if_missing(MODEL_PATH, MODEL_URL)
                _download_if_missing(LABELS_PATH, LABELS_URL)
                _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
                _labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return _session, _labels or []


def _preprocess(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((int(round(width * scale)), int(round(height * scale))))
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(cropped).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)
    return array.astype("float32")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def run_inference(image_path: str | Path) -> InferenceResult:
    path = Path(image_path)
    try:
        session, labels = _load_resources()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logits = session.run([output_name], {input_name: _preprocess(path)})[0][0]
        probs = _softmax(np.asarray(logits, dtype="float32"))
        top_indices = np.argsort(probs)[::-1][:5]

        detections: list[dict[str, Any]] = []
        chosen = "unknown"
        chosen_score = float(probs[top_indices[0]])
        for idx in top_indices:
            raw_label = labels[int(idx)] if int(idx) < len(labels) else f"class-{idx}"
            mapped = IMAGENET_TO_CLASS.get(raw_label, "unknown")
            score = float(probs[int(idx)])
            detections.append(
                {
                    "class_name": mapped,
                    "label": CLASS_META[mapped]["display"],
                    "score": score,
                    "box": [0, 0, 0, 0],
                    "raw_label": raw_label,
                }
            )
            if mapped != "unknown" and chosen == "unknown":
                chosen = mapped
                chosen_score = score

        return InferenceResult(
            predicted_class=chosen,
            predicted_label=CLASS_META[chosen]["label"],
            confidence=chosen_score,
            model_mode="mobilenetv2-onnx-imagenet-mapping",
            detections=detections,
            annotated_path=None,
        )
    except Exception as exc:
        raise EngineUnavailable(f"推理引擎初始化或执行失败：{exc}") from exc
