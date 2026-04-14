#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import ssl
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "experiments" / "demo_dataset"
ROBUSTNESS_DIR = DATA_DIR / "experiments" / "robustness"
MANIFEST_PATH = DATA_DIR / "experiments" / "dataset_manifest.json"
EXTERNAL_DIR = DATA_DIR / "external" / "coco"
IMAGE_CACHE_ROOT = EXTERNAL_DIR / "images"
ANNOTATIONS_ZIP_PATH = EXTERNAL_DIR / "annotations_trainval2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URLS = {
    "train2017": "http://images.cocodataset.org/train2017/{file_name}",
    "val2017": "http://images.cocodataset.org/val2017/{file_name}",
}
IMAGE_SIZE = (256, 256)
SEED = 20260325

CLASS_SPECS = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "truck": 8,
    "traffic_light": 10,
}

SPLIT_COUNTS = {
    "train": 64,
    "val": 16,
    "test": 16,
}

SCENARIOS = [
    ("normal", {"brightness": 1.0, "noise": 0.0, "rotate": 0, "crop_ratio": 1.0, "blur": 0.0}),
    ("low_light", {"brightness": 0.35, "noise": 0.02, "rotate": -5, "crop_ratio": 0.92, "blur": 0.6}),
    ("gaussian_noise", {"brightness": 0.9, "noise": 0.09, "rotate": 6, "crop_ratio": 0.95, "blur": 0.0}),
    ("partial_crop", {"brightness": 1.0, "noise": 0.02, "rotate": 10, "crop_ratio": 0.72, "blur": 0.4}),
]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def add_noise(image: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    if strength <= 0:
        return image
    array = np.asarray(image).astype("float32") / 255.0
    noise = rng.normalvariate(0, strength)
    noise_map = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, strength, array.shape)
    array = np.clip(array + noise_map + noise * 0.05, 0.0, 1.0)
    return Image.fromarray((array * 255).astype("uint8"))


def crop_with_ratio(image: Image.Image, ratio: float, rng: random.Random) -> Image.Image:
    if ratio >= 0.999:
        return image
    width, height = image.size
    new_w = max(32, int(width * ratio))
    new_h = max(32, int(height * ratio))
    max_x = max(0, width - new_w)
    max_y = max(0, height - new_h)
    left = rng.randint(0, max_x) if max_x else 0
    top = rng.randint(0, max_y) if max_y else 0
    image = image.crop((left, top, left + new_w, top + new_h))
    return image.resize(IMAGE_SIZE, Image.BILINEAR)


def apply_recipe(image: Image.Image, recipe: dict, rng: random.Random) -> Image.Image:
    result = image.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    if recipe.get("flip"):
        result = result.transpose(Image.FLIP_LEFT_RIGHT)
    rotate = float(recipe.get("rotate", 0))
    if rotate:
        result = result.rotate(rotate, resample=Image.BILINEAR)
    result = crop_with_ratio(result, float(recipe.get("crop_ratio", 1.0)), rng)
    if recipe.get("brightness", 1.0) != 1.0:
        result = ImageEnhance.Brightness(result).enhance(float(recipe["brightness"]))
    if recipe.get("contrast", 1.0) != 1.0:
        result = ImageEnhance.Contrast(result).enhance(float(recipe["contrast"]))
    if recipe.get("color", 1.0) != 1.0:
        result = ImageEnhance.Color(result).enhance(float(recipe["color"]))
    blur = float(recipe.get("blur", 0.0))
    if blur > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur))
    result = add_noise(result, float(recipe.get("noise", 0.0)), rng)
    return result


def stable_offset(*parts: str) -> int:
    joined = "|".join(parts)
    return sum(ord(ch) for ch in joined)


def random_recipe(split: str, rng: random.Random) -> dict:
    base = {
        "flip": rng.random() < 0.5,
        "rotate": rng.uniform(-15, 15),
        "crop_ratio": rng.uniform(0.76, 1.0),
        "brightness": rng.uniform(0.55, 1.25),
        "contrast": rng.uniform(0.75, 1.25),
        "color": rng.uniform(0.8, 1.2),
        "noise": rng.uniform(0.0, 0.06),
        "blur": rng.uniform(0.0, 1.0),
    }
    if split == "val":
        base["rotate"] = rng.uniform(-8, 8)
        base["noise"] = rng.uniform(0.0, 0.03)
        base["crop_ratio"] = rng.uniform(0.85, 1.0)
    if split == "test":
        base["brightness"] = rng.uniform(0.35, 1.15)
        base["noise"] = rng.uniform(0.02, 0.10)
        base["crop_ratio"] = rng.uniform(0.68, 0.96)
        base["blur"] = rng.uniform(0.0, 1.4)
    return base


def download_file(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    req = urllib.request.Request(url, headers={"User-Agent": "openclaw"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=180, context=context) as response, open(target_path, "wb") as output:
        shutil.copyfileobj(response, output)
    return target_path


def load_coco_annotations() -> dict[str, dict]:
    download_file(ANNOTATIONS_URL, ANNOTATIONS_ZIP_PATH)
    payloads = {}
    with zipfile.ZipFile(ANNOTATIONS_ZIP_PATH) as zf:
        for split_name in ("train2017", "val2017"):
            with zf.open(f"annotations/instances_{split_name}.json") as f:
                payloads[split_name] = json.load(f)
    return payloads


def valid_annotation(ann: dict) -> bool:
    if ann.get("iscrowd"):
        return False
    x, y, w, h = ann["bbox"]
    if min(w, h) < 48:
        return False
    if ann.get("area", w * h) < 2500:
        return False
    return True


def select_annotations(coco_payloads: dict[str, dict]) -> dict[str, dict[str, list[dict]]]:
    rng = random.Random(SEED)
    target_ids = set(CLASS_SPECS.values())
    per_class: dict[str, list[dict]] = {label: [] for label in CLASS_SPECS}
    id_to_label = {value: key for key, value in CLASS_SPECS.items()}

    for source_split, coco in coco_payloads.items():
        images = {item["id"]: item for item in coco["images"]}
        categories = {item["id"]: item["name"] for item in coco["categories"]}
        for ann in coco["annotations"]:
            category_id = ann["category_id"]
            if category_id not in target_ids or not valid_annotation(ann):
                continue
            image = images.get(ann["image_id"])
            if not image:
                continue
            label = id_to_label[category_id]
            per_class[label].append(
                {
                    "annotation_id": f"{source_split}:{ann['id']}",
                    "image_id": ann["image_id"],
                    "label": label,
                    "coco_label": categories[category_id],
                    "bbox": ann["bbox"],
                    "area": ann.get("area"),
                    "file_name": image["file_name"],
                    "width": image["width"],
                    "height": image["height"],
                    "source_split": source_split,
                    "source_url": IMAGE_URLS[source_split].format(file_name=image["file_name"]),
                }
            )

    for items in per_class.values():
        rng.shuffle(items)

    selections: dict[str, dict[str, list[dict]]] = {split: {label: [] for label in CLASS_SPECS} for split in SPLIT_COUNTS}
    used_annotations: set[str] = set()
    for split, count in SPLIT_COUNTS.items():
        for label, candidates in per_class.items():
            chosen: list[dict] = []
            for candidate in candidates:
                if candidate["annotation_id"] in used_annotations:
                    continue
                chosen.append(candidate)
                used_annotations.add(candidate["annotation_id"])
                if len(chosen) >= count:
                    break
            if len(chosen) < count:
                raise SystemExit(f"{label} 在 {split} 中可用样本不足：需要 {count}，实际 {len(chosen)}")
            selections[split][label] = chosen
    return selections


def fetch_image(item: dict) -> Path:
    source_split = item["source_split"]
    file_name = item["file_name"]
    path = IMAGE_CACHE_ROOT / source_split / file_name
    if path.exists() and path.stat().st_size > 0:
        return path
    return download_file(IMAGE_URLS[source_split].format(file_name=file_name), path)


def prefetch_images(selections: dict[str, dict[str, list[dict]]]) -> None:
    unique_items = {}
    for split_map in selections.values():
        for items in split_map.values():
            for item in items:
                unique_items[(item["source_split"], item["file_name"])] = item
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_image, item): key for key, item in unique_items.items()}
        for future in as_completed(futures):
            future.result()


def clamp(value: float, low: int, high: int) -> int:
    return max(low, min(int(round(value)), high))


def crop_annotation(image: Image.Image, bbox: Iterable[float]) -> Image.Image:
    x, y, w, h = bbox
    img_w, img_h = image.size
    margin_x = w * 0.12
    margin_y = h * 0.12
    left = clamp(x - margin_x, 0, img_w - 1)
    top = clamp(y - margin_y, 0, img_h - 1)
    right = clamp(x + w + margin_x, left + 1, img_w)
    bottom = clamp(y + h + margin_y, top + 1, img_h)
    return image.crop((left, top, right, bottom)).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)


def save_split_samples(selections: dict[str, dict[str, list[dict]]], manifest: list[dict]) -> None:
    for split, label_map in selections.items():
        for label, items in label_map.items():
            split_dir = OUTPUT_DIR / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(items):
                image_path = fetch_image(item)
                with Image.open(image_path) as image:
                    sample = crop_annotation(image, item["bbox"])
                filename = f"{label}_{split}_{index:03d}.jpg"
                output_path = split_dir / filename
                sample.save(output_path, quality=92)
                manifest.append(
                    {
                        "split": split,
                        "label": label,
                        "path": str(output_path.relative_to(ROOT)),
                        "source": {
                            "dataset": f"COCO 2017 {item['source_split']}",
                            "image_file": item["file_name"],
                            "source_url": item["source_url"],
                            "annotation_id": item["annotation_id"],
                            "image_id": item["image_id"],
                            "bbox": item["bbox"],
                            "area": item["area"],
                            "coco_label": item["coco_label"],
                        },
                    }
                )


def save_robustness_samples(manifest: list[dict]) -> None:
    ensure_clean_dir(ROBUSTNESS_DIR)
    for label in CLASS_SPECS:
        sample_candidates = sorted((OUTPUT_DIR / "test" / label).glob("*.jpg"))
        if not sample_candidates:
            continue
        with Image.open(sample_candidates[0]) as opened:
            source = opened.convert("RGB")
        label_dir = ROBUSTNESS_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for scenario_name, recipe in SCENARIOS:
            rng = random.Random(SEED + stable_offset(label, scenario_name))
            sample = apply_recipe(source.copy(), recipe, rng)
            filename = f"{label}_{scenario_name}.jpg"
            output_path = label_dir / filename
            sample.save(output_path, quality=92)
            manifest.append(
                {
                    "split": "robustness",
                    "label": label,
                    "scenario": scenario_name,
                    "path": str(output_path.relative_to(ROOT)),
                    "recipe": recipe,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建公开 COCO 七类道路目标裁剪数据集")
    parser.add_argument("--train-per-class", type=int, default=SPLIT_COUNTS["train"])
    parser.add_argument("--val-per-class", type=int, default=SPLIT_COUNTS["val"])
    parser.add_argument("--test-per-class", type=int, default=SPLIT_COUNTS["test"])
    return parser.parse_args()


def main() -> None:
    global SPLIT_COUNTS
    args = parse_args()
    SPLIT_COUNTS = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }

    ensure_clean_dir(OUTPUT_DIR)
    (DATA_DIR / "experiments").mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    coco_payloads = load_coco_annotations()
    selections = select_annotations(coco_payloads)
    prefetch_images(selections)

    manifest: list[dict] = []
    save_split_samples(selections, manifest)
    save_robustness_samples(manifest)

    summary = {
        "dataset_name": "coco-road7-crops",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "classes": list(CLASS_SPECS.keys()),
        "split_counts_per_class": SPLIT_COUNTS,
        "total_generated": len(manifest),
        "source": {
            "name": "COCO 2017 object crops",
            "annotations_zip": str(ANNOTATIONS_ZIP_PATH.relative_to(ROOT)),
            "annotations_url": ANNOTATIONS_URL,
            "image_splits": ["train2017", "val2017"],
        },
        "items": manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "manifest": str(MANIFEST_PATH.relative_to(ROOT)), "total_generated": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
