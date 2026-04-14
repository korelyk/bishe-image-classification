#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision import datasets, transforms

from app.vision_models import build_model, model_parameter_stats

DATASET_DIR = ROOT / "data" / "experiments" / "demo_dataset"
ROBUSTNESS_DIR = ROOT / "data" / "experiments" / "robustness"
CHECKPOINT_DIR = ROOT / "data" / "experiments" / "checkpoints"
REPORT_DIR = ROOT / "data" / "reports"
DOCS_DIR = ROOT / "docs"
MANIFEST_PATH = ROOT / "data" / "experiments" / "dataset_manifest.json"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b0"]
DISPLAY = {
    "resnet50": "ResNet50",
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}
NOTES = {
    "resnet50": "特征表达强，参数量较大",
    "mobilenet_v2": "轻量化明显，推理速度最快",
    "efficientnet_b0": "精度与速度较均衡",
}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"dataset_name": "unknown", "classes": [], "split_counts_per_class": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_dataset(path: Path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return datasets.ImageFolder(path, transform=transform)


def evaluate_dataset(model, dataset, device):
    model.eval()
    correct = 0
    total = 0
    timings = []
    with torch.inference_mode():
        for image, target in dataset:
            input_tensor = image.unsqueeze(0).to(device)
            start = time.perf_counter()
            output = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            timings.append(elapsed)
            pred = int(output.argmax(dim=1).item())
            correct += int(pred == target)
            total += 1
    accuracy = (correct / total * 100) if total else 0.0
    avg_ms = (sum(timings) / len(timings)) if timings else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return round(accuracy, 2), round(avg_ms, 2), round(fps, 2)


def load_checkpoint(model_name: str, device: torch.device):
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model, _ = build_model(model_name, num_classes=len(class_names), pretrained=True, freeze_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return checkpoint, model


def load_training_report(model_name: str) -> dict:
    path = REPORT_DIR / f"training_{model_name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def benchmark_all() -> tuple[dict, dict]:
    manifest = load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = load_dataset(DATASET_DIR / "test")
    val_dataset = load_dataset(DATASET_DIR / "val")
    train_dataset = load_dataset(DATASET_DIR / "train")
    robustness_summary: dict[str, list[dict]] = {}
    rows = []

    for model_name in MODELS:
        checkpoint, model = load_checkpoint(model_name, device)
        training_report = load_training_report(model_name)
        val_acc, _, _ = evaluate_dataset(model, val_dataset, device)
        test_acc, avg_ms, fps = evaluate_dataset(model, test_dataset, device)
        stats = model_parameter_stats(model)

        rows.append(
            {
                "model_name": model_name,
                "display_name": DISPLAY[model_name],
                "parameters_million": round(stats["total"] / 1_000_000, 2),
                "trainable_million": round(stats["trainable"] / 1_000_000, 2),
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "avg_inference_ms": avg_ms,
                "fps": fps,
                "device": str(device),
                "checkpoint": str((CHECKPOINT_DIR / f"{model_name}.pt").relative_to(ROOT)),
                "training_seconds": training_report.get("training_seconds"),
                "best_val_acc_from_training": training_report.get("best_val_acc"),
                "note": NOTES[model_name],
                "num_classes": len(checkpoint["class_names"]),
            }
        )

        robustness_dataset = datasets.ImageFolder(
            ROBUSTNESS_DIR,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ]
            ),
        )
        with torch.inference_mode():
            for (path_str, target_idx) in robustness_dataset.samples:
                image, _ = robustness_dataset.loader(path_str), target_idx
                image = robustness_dataset.transform(image)
                output = model(image.unsqueeze(0).to(device))
                pred = int(output.argmax(dim=1).item())
                label = robustness_dataset.classes[target_idx]
                stem = Path(path_str).stem
                scenario = stem.removeprefix(f"{label}_")
                robustness_summary.setdefault(scenario, []).append(
                    {
                        "model_name": model_name,
                        "correct": int(pred == target_idx),
                        "label": label,
                    }
                )

    best_accuracy = max(rows, key=lambda item: (item["test_accuracy"], -item["avg_inference_ms"]))
    fastest = min(rows, key=lambda item: item["avg_inference_ms"])

    comparison_payload = {
        "dataset_name": manifest.get("dataset_name", "unknown"),
        "classes": manifest.get("classes", []),
        "split_counts_per_class": manifest.get("split_counts_per_class", {}),
        "dataset_size": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_accuracy_model": best_accuracy["display_name"],
        "fastest_model": fastest["display_name"],
        "models": rows,
        "note": "当前实验基于公开 COCO 目标检测数据裁剪出的七类道路目标分类集，已能用于展示真实公开数据来源、训练流程、模型对比和论文实验方法。若后续拿到更贴近学校题目的专用七类数据，可直接复用现有脚本重跑。",
    }

    robustness_items = []
    for scenario, items in robustness_summary.items():
        stats_by_model = {}
        for item in items:
            stats_by_model.setdefault(item["model_name"], {"correct": 0, "total": 0})
            stats_by_model[item["model_name"]]["correct"] += item["correct"]
            stats_by_model[item["model_name"]]["total"] += 1
        ranked = sorted(
            (
                {
                    "model_name": model_name,
                    "accuracy": round(values["correct"] / values["total"] * 100, 2),
                }
                for model_name, values in stats_by_model.items()
            ),
            key=lambda x: x["accuracy"],
            reverse=True,
        )
        best = ranked[0]
        robustness_items.append(
            {
                "scene": scenario,
                "best_model": DISPLAY[best["model_name"]],
                "summary": " / ".join(f"{DISPLAY[row['model_name']]} {row['accuracy']}%" for row in ranked),
            }
        )

    robustness_payload = {
        "dataset": "robustness-mini-suite",
        "items": robustness_items,
    }
    return comparison_payload, robustness_payload


def write_markdown(comparison: dict, robustness: dict) -> None:
    class_text = " / ".join(comparison.get("classes", [])) or "未知"
    lines = [
        "# 多模型实验对比结果",
        "",
        f"- 数据集：{comparison['dataset_name']}",
        f"- 类别：{class_text}",
        f"- 训练/验证/测试样本量：{comparison['dataset_size']['train']} / {comparison['dataset_size']['val']} / {comparison['dataset_size']['test']}",
        f"- 准确率最优模型：{comparison['best_accuracy_model']}",
        f"- 推理速度最优模型：{comparison['fastest_model']}",
        "",
        "## 1. 主实验结果",
        "",
        "| 模型 | 参数量(M) | 类别数 | 验证准确率 | 测试准确率 | 单张耗时(ms) | FPS | 备注 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison["models"]:
        lines.append(
            f"| {row['display_name']} | {row['parameters_million']} | {row['num_classes']} | {row['val_accuracy']}% | {row['test_accuracy']}% | {row['avg_inference_ms']} | {row['fps']} | {row['note']} |"
        )
    lines.extend(["", "## 2. 复杂场景鲁棒性", ""])
    for item in robustness["items"]:
        lines.append(f"- **{item['scene']}**：{item['summary']}，最佳模型为 **{item['best_model']}**。")
    lines.extend(["", "## 3. 说明", "", f"> {comparison['note']}"])
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison, robustness = benchmark_all()
    (REPORT_DIR / "model_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "robustness_report.json").write_text(json.dumps(robustness, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(comparison, robustness)
    print(json.dumps({"success": True, "comparison": "data/reports/model_comparison.json", "robustness": "data/reports/robustness_report.json"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
