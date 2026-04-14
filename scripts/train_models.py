#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.vision_models import AddGaussianNoise, build_model, model_parameter_stats
DATASET_DIR = ROOT / "data" / "experiments" / "demo_dataset"
OUTPUT_DIR = ROOT / "data" / "experiments" / "checkpoints"
REPORT_DIR = ROOT / "data" / "reports"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(batch_size: int) -> tuple[dict[str, datasets.ImageFolder], dict[str, DataLoader]]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.35, contrast=0.25, saturation=0.2),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.04),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    datasets_map = {
        "train": datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform),
        "val": datasets.ImageFolder(DATASET_DIR / "val", transform=eval_transform),
        "test": datasets.ImageFolder(DATASET_DIR / "test", transform=eval_transform),
    }
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
        for split, dataset in datasets_map.items()
    }
    return datasets_map, loaders


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    running_loss = 0.0
    running_correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if train:
            loss.backward()
            optimizer.step()
        preds = outputs.argmax(dim=1)
        running_loss += float(loss.item()) * inputs.size(0)
        running_correct += int((preds == targets).sum().item())
        total += int(inputs.size(0))
    avg_loss = running_loss / max(total, 1)
    avg_acc = running_correct / max(total, 1)
    return avg_loss, avg_acc


def train_one_model(model_name: str, epochs: int, batch_size: int, lr: float, freeze_backbone: bool, seed: int) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_map, loaders = make_loaders(batch_size)
    class_names = datasets_map["train"].classes
    model, _ = build_model(model_name, num_classes=len(class_names), pretrained=True, freeze_backbone=freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
    best_state = None
    best_val_acc = -1.0
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc * 100, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc * 100, 2),
            }
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    elapsed = time.perf_counter() - started
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_DIR / f"{model_name}.pt"
    torch.save(
        {
            "model_name": model_name,
            "class_names": class_names,
            "class_to_idx": datasets_map["train"].class_to_idx,
            "state_dict": best_state,
            "best_val_acc": float(best_val_acc),
            "freeze_backbone": freeze_backbone,
        },
        checkpoint_path,
    )

    payload = {
        "model_name": model_name,
        "class_names": class_names,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "freeze_backbone": freeze_backbone,
        "device": str(device),
        "best_val_acc": round(best_val_acc * 100, 2),
        "training_seconds": round(elapsed, 2),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "history": history,
        "parameter_stats": model_parameter_stats(model),
    }
    (REPORT_DIR / f"training_{model_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_name": model_name, "best_val_acc": payload["best_val_acc"], "checkpoint": payload["checkpoint"]}, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练多模型道路目标演示集")
    parser.add_argument("--models", nargs="+", default=["resnet50", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--no-freeze", action="store_true", help="取消冻结骨干网络")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATASET_DIR.exists():
        raise SystemExit(f"未找到数据集目录：{DATASET_DIR}，请先运行 scripts/build_demo_dataset.py")
    summary = []
    for model_name in args.models:
        summary.append(
            train_one_model(
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                freeze_backbone=not args.no_freeze,
                seed=args.seed,
            )
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
