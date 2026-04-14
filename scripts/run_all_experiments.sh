#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
python scripts/build_demo_dataset.py
python scripts/train_models.py --epochs 8 --batch-size 8
python scripts/benchmark_models.py
python scripts/generate_gradcam_gallery.py
