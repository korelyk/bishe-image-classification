#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_engine import run_inference
SAMPLES = [
    ROOT / "data" / "demo" / "bike_demo.jpg",
    ROOT / "data" / "demo" / "bus_demo.jpg",
    ROOT / "data" / "demo" / "car_demo.jpg",
]
MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b0"]
REPORT_DIR = ROOT / "data" / "reports"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for sample in SAMPLES:
        for model_name in MODELS:
            result = run_inference(sample, model_name=model_name, with_gradcam=True)
            items.append(
                {
                    "sample": str(sample.relative_to(ROOT)),
                    "model_name": model_name,
                    "model_display_name": result.model_display_name,
                    "predicted_label": result.predicted_label,
                    "confidence": round(result.confidence, 4),
                    "gradcam_path": result.annotated_path,
                }
            )
    payload = {"success": True, "items": items}
    (REPORT_DIR / "gradcam_gallery.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "count": len(items), "report": "data/reports/gradcam_gallery.json"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
