from __future__ import annotations

import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import init_db, insert_prediction, list_predictions, stats
from .model_engine import CLASS_META, EngineUnavailable, run_inference

APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_ROOT / "app" / "static"
TEMPLATE_DIR = APP_ROOT / "app" / "templates"
UPLOAD_DIR = APP_ROOT / "data" / "uploads"

app = FastAPI(title="基于深度学习的图片分类系统", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/files", StaticFiles(directory=APP_ROOT / "data"), name="files")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


@app.on_event("startup")
def on_startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


def _admin_guard(x_admin_password: Annotated[Optional[str], Header()] = None) -> None:
    expected = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not expected:
        return
    if x_admin_password != expected:
        raise HTTPException(status_code=401, detail="管理员口令错误")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "project_name": "基于深度学习的图片分类系统",
            "class_meta": CLASS_META,
        },
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "admin.html",
        {"project_name": "基于深度学习的图片分类系统"},
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "service": "graduation-image-classification-system",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    target = UPLOAD_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(8)}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    target.write_bytes(content)

    try:
        result = run_inference(target)
    except EngineUnavailable as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = {
        "filename": file.filename or target.name,
        "original_path": str(target.relative_to(APP_ROOT)),
        "annotated_path": result.annotated_path,
        "predicted_class": result.predicted_class,
        "predicted_label": result.predicted_label,
        "confidence": round(result.confidence, 4),
        "model_mode": result.model_mode,
        "raw_json": json.dumps(result.detections, ensure_ascii=False),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    record_id = insert_prediction(payload)

    return JSONResponse(
        {
            "success": True,
            "id": record_id,
            "filename": payload["filename"],
            "predicted_class": result.predicted_class,
            "predicted_label": result.predicted_label,
            "confidence": round(result.confidence, 4),
            "model_mode": result.model_mode,
            "detections": result.detections,
            "image_url": f"/files/{payload['original_path']}",
            "annotated_url": f"/files/{payload['annotated_path']}" if payload["annotated_path"] else None,
            "created_at": payload["created_at"],
        }
    )


@app.get("/api/history")
def history(limit: int = 20) -> dict:
    items = list_predictions(limit)
    for item in items:
        item["image_url"] = f"/files/{item['original_path']}"
        item["annotated_url"] = f"/files/{item['annotated_path']}" if item.get("annotated_path") else None
        try:
            item["detections"] = json.loads(item["raw_json"])
        except Exception:
            item["detections"] = []
    return {"success": True, "items": items}


@app.get("/api/admin/stats")
def admin_stats(_: None = Depends(_admin_guard)) -> dict:
    recent = list_predictions(10)
    for item in recent:
        item["image_url"] = f"/files/{item['original_path']}"
        item["annotated_url"] = f"/files/{item['annotated_path']}" if item.get("annotated_path") else None
    return {"success": True, "stats": stats(), "recent": recent}
