from __future__ import annotations

import json
import os
import secrets
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .db import delete_prediction, get_prediction, init_db, insert_prediction, list_predictions, stats
from .model_engine import (
    CLASS_META,
    EngineUnavailable,
    get_available_models,
    run_inference,
    run_stress_test,
)

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
STATIC_DIR = APP_ROOT / "app" / "static"
TEMPLATE_DIR = APP_ROOT / "app" / "templates"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORTS_DIR = DATA_DIR / "reports"
DEMO_DIR = DATA_DIR / "demo"
PROJECT_NAME = "基于深度学习的图片分类系统"

APP_VERSION = "2.4.1"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FILE_ROOTS = ("uploads", "annotated", "stress", "demo")

NAV_ITEMS = [
    {"key": "home", "label": "系统总览", "href": "/"},
    {"key": "single", "label": "单图识别", "href": "/single"},
    {"key": "scenarios", "label": "复杂场景", "href": "/scenarios"},
    {"key": "reports", "label": "评估报告", "href": "/reports"},
    {"key": "admin", "label": "后台管理", "href": "/admin"},
]

PAGE_TITLES = {
    "home": "系统总览",
    "single": "单图识别",
    "scenarios": "复杂场景",
    "reports": "评估报告",
    "admin": "后台管理",
}

DEMO_SAMPLES = [
    {
        "name": "car_demo",
        "filename": "car_demo.jpg",
        "title": "汽车样例",
        "description": "适合演示基础识别与 Grad-CAM 热力图。",
    },
    {
        "name": "bus_demo",
        "filename": "bus_demo.jpg",
        "title": "公交车样例",
        "description": "适合展示类别区分与复杂背景下的识别结果。",
    },
    {
        "name": "bike_demo",
        "filename": "bike_demo.jpg",
        "title": "自行车样例",
        "description": "适合展示小目标识别与模型关注区域。",
    },
]

app = FastAPI(title=PROJECT_NAME, version=APP_VERSION)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


def _relative_data_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = str(path).replace("\\", "/").lstrip("/")
    if normalized.startswith("data/"):
        normalized = normalized[len("data/") :]
    if not normalized or ".." in Path(normalized).parts:
        return None
    return normalized


def _files_url(path: Optional[str]) -> Optional[str]:
    normalized = _relative_data_path(path)
    if not normalized:
        return None
    return f"/files/{normalized}"


def _data_path(path: Optional[str]) -> Optional[Path]:
    normalized = _relative_data_path(path)
    if not normalized:
        return None
    candidate = (DATA_DIR / normalized).resolve()
    data_root = DATA_DIR.resolve()
    if data_root == candidate or data_root in candidate.parents:
        return candidate
    return None


def _allowed_public_data_path(path: Optional[str]) -> Optional[Path]:
    normalized = _relative_data_path(path)
    if not normalized:
        return None
    parts = Path(normalized).parts
    if not parts or parts[0] not in ALLOWED_FILE_ROOTS:
        return None
    candidate = _data_path(normalized)
    if candidate and candidate.exists() and candidate.is_file():
        return candidate
    return None


def _validate_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip()
    allowed = {item["name"] for item in get_available_models()}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的模型：{model_name}")
    return normalized


def _validate_upload(file: UploadFile, content: bytes) -> None:
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="单张图片不能超过 8MB")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG、WEBP 图片")

    try:
        Image.open(BytesIO(content)).verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="上传内容不是有效图片") from exc


def _load_report_json(filename: str) -> dict[str, Any]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {"success": False, "message": f"报告文件不存在：{filename}"}
    try:
        return {"success": True, "data": json.loads(path.read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"success": False, "message": f"读取报告失败：{exc}"}


@app.on_event("startup")
def on_startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


def _admin_guard(x_admin_password: Annotated[Optional[str], Header()] = None) -> None:
    expected = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not expected:
        return
    if x_admin_password != expected:
        raise HTTPException(status_code=401, detail="管理员口令错误")


def _store_upload(file: UploadFile, content: bytes) -> Path:
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        suffix = ".jpg"
    target = UPLOAD_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(8)}{suffix}"
    target.write_bytes(content)
    return target


def _save_prediction_result(filename: str, target: Path, result: Any) -> tuple[int, dict[str, Any]]:
    payload = {
        "filename": filename or target.name,
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
    return record_id, payload


def _save_prediction_record(file: UploadFile, target: Path, result: Any) -> tuple[int, dict[str, Any]]:
    return _save_prediction_result(file.filename or target.name, target, result)


def _response_from_prediction(record_id: int, payload: dict[str, Any], result: Any) -> dict[str, Any]:
    return {
        "success": True,
        "id": record_id,
        "filename": payload["filename"],
        "predicted_class": result.predicted_class,
        "predicted_label": result.predicted_label,
        "confidence": round(result.confidence, 4),
        "model_mode": result.model_mode,
        "model_name": result.model_name,
        "model_display_name": result.model_display_name,
        "gradcam_enabled": result.gradcam_enabled,
        "detections": result.detections,
        "image_url": _files_url(payload["original_path"]),
        "annotated_url": _files_url(payload["annotated_path"]),
        "created_at": payload["created_at"],
    }


def _response_from_history(item: dict[str, Any]) -> dict[str, Any]:
    item["image_url"] = _files_url(item["original_path"])
    item["annotated_url"] = _files_url(item.get("annotated_path"))
    try:
        item["detections"] = json.loads(item["raw_json"])
    except Exception:
        item["detections"] = []
    return item


def _template_context(*, active_page: str) -> dict[str, Any]:
    demo_samples = []
    for item in DEMO_SAMPLES:
        demo_samples.append(
            {
                **item,
                "image_url": _files_url(str((DEMO_DIR / item["filename"]).relative_to(APP_ROOT))),
            }
        )
    return {
        "project_name": PROJECT_NAME,
        "active_page": active_page,
        "page_title": PAGE_TITLES.get(active_page, PROJECT_NAME),
        "nav_items": NAV_ITEMS,
        "class_meta": CLASS_META,
        "model_options": get_available_models(),
        "demo_samples": demo_samples,
        "app_version": APP_VERSION,
    }


def _sample_target(sample_name: str) -> tuple[dict[str, Any], Path]:
    sample = next((item for item in DEMO_SAMPLES if item["name"] == sample_name), None)
    if not sample:
        raise HTTPException(status_code=404, detail="样例不存在")

    target = (DEMO_DIR / sample["filename"]).resolve()
    demo_root = DEMO_DIR.resolve()
    if demo_root not in target.parents or not target.exists():
        raise HTTPException(status_code=404, detail="样例文件不存在")
    return sample, target


def _is_deletable_data_file(target: Path) -> bool:
    allowed_roots = [
        UPLOAD_DIR.resolve(),
        (DATA_DIR / "annotated").resolve(),
        (DATA_DIR / "stress").resolve(),
    ]
    return any(root == target or root in target.parents for root in allowed_roots)


@app.get("/files/{file_path:path}")
def serve_data_file(file_path: str) -> FileResponse:
    target = _allowed_public_data_path(file_path)
    if not target:
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        _template_context(active_page="home"),
    )


@app.get("/single", response_class=HTMLResponse)
def single_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "single.html",
        _template_context(active_page="single"),
    )


@app.get("/scenarios", response_class=HTMLResponse)
def scenarios_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "scenarios.html",
        _template_context(active_page="scenarios"),
    )


@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "reports.html",
        _template_context(active_page="reports"),
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "admin.html",
        _template_context(active_page="admin"),
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "service": "graduation-image-classification-system",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": APP_VERSION,
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"success": True, "items": get_available_models()}


@app.get("/api/reports/model-comparison")
def model_comparison_report() -> dict[str, Any]:
    return _load_report_json("model_comparison.json")


@app.get("/api/reports/robustness")
def robustness_report() -> dict[str, Any]:
    return _load_report_json("robustness_report.json")


@app.get("/api/reports/training-summary")
def training_summary_report() -> dict[str, Any]:
    return _load_report_json("training_summary.json")


@app.get("/api/reports/gradcam-gallery")
def gradcam_gallery_report() -> dict[str, Any]:
    return _load_report_json("gradcam_gallery.json")


@app.post("/api/classify")
async def classify(
    file: UploadFile = File(...),
    model_name: str = Form("resnet50"),
    with_gradcam: bool = Form(True),
) -> JSONResponse:
    model_name = _validate_model_name(model_name)
    content = await file.read()
    _validate_upload(file, content)

    target = _store_upload(file, content)
    try:
        result = await run_in_threadpool(
            run_inference,
            target,
            model_name=model_name,
            with_gradcam=with_gradcam,
        )
    except EngineUnavailable as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record_id, payload = _save_prediction_record(file, target, result)
    return JSONResponse(_response_from_prediction(record_id, payload, result))


@app.post("/api/classify/sample")
def classify_sample(
    sample_name: str = Form(...),
    model_name: str = Form("resnet50"),
    with_gradcam: bool = Form(True),
) -> JSONResponse:
    model_name = _validate_model_name(model_name)
    sample, target = _sample_target(sample_name)
    try:
        result = run_inference(target, model_name=model_name, with_gradcam=with_gradcam)
    except EngineUnavailable as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record_id, payload = _save_prediction_result(sample["filename"], target, result)
    response = _response_from_prediction(record_id, payload, result)
    response["sample_name"] = sample["name"]
    response["sample_title"] = sample["title"]
    return JSONResponse(response)


@app.post("/api/classify/batch")
async def classify_batch(
    files: list[UploadFile] = File(...),
    model_name: str = Form("resnet50"),
    with_gradcam: bool = Form(False),
) -> JSONResponse:
    model_name = _validate_model_name(model_name)
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传 1 张图片")
    if len(files) > 12:
        raise HTTPException(status_code=400, detail="单次批量识别最多支持 12 张图片")

    items: list[dict[str, Any]] = []
    for file in files:
        content = await file.read()
        try:
            _validate_upload(file, content)
        except HTTPException as exc:
            items.append({"success": False, "filename": file.filename, "detail": exc.detail})
            continue

        target = _store_upload(file, content)
        try:
            result = await run_in_threadpool(
                run_inference,
                target,
                model_name=model_name,
                with_gradcam=with_gradcam,
            )
        except EngineUnavailable as exc:
            items.append({"success": False, "filename": file.filename, "detail": str(exc)})
            continue

        record_id, payload = _save_prediction_record(file, target, result)
        items.append(_response_from_prediction(record_id, payload, result))

    return JSONResponse(
        {
            "success": True,
            "count": len(items),
            "items": items,
            "model_name": model_name,
            "with_gradcam": with_gradcam,
        }
    )


@app.post("/api/classify/stress-test")
async def classify_stress_test(
    file: UploadFile = File(...),
    model_name: str = Form("resnet50"),
) -> JSONResponse:
    model_name = _validate_model_name(model_name)
    content = await file.read()
    _validate_upload(file, content)

    target = _store_upload(file, content)
    try:
        items = await run_in_threadpool(run_stress_test, target, model_name=model_name)
    except EngineUnavailable as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = []
    for item in items:
        payload.append(
            {
                **item,
                "image_url": _files_url(item["image_path"]),
            }
        )

    return JSONResponse(
        {
            "success": True,
            "source_image_url": _files_url(str(target.relative_to(APP_ROOT))),
            "model_name": model_name,
            "count": len(payload),
            "items": payload,
        }
    )


@app.get("/api/history")
def history(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(limit, 100))
    items = [_response_from_history(item) for item in list_predictions(limit)]
    return {"success": True, "items": items}


@app.get("/api/admin/stats")
def admin_stats(_: None = Depends(_admin_guard)) -> dict[str, Any]:
    recent = [_response_from_history(item) for item in list_predictions(10)]
    return {"success": True, "stats": stats(), "recent": recent}


@app.delete("/api/admin/predictions/{record_id}")
def admin_delete_prediction(record_id: int, _: None = Depends(_admin_guard)) -> dict[str, Any]:
    record = get_prediction(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")

    removed_files: list[str] = []
    for raw_path in [record.get("original_path"), record.get("annotated_path")]:
        target = _data_path(raw_path)
        if target and target.exists() and target.is_file() and _is_deletable_data_file(target):
            try:
                target.unlink()
                removed_files.append(str(target.relative_to(APP_ROOT)))
            except Exception:
                pass

    deleted = delete_prediction(record_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="删除记录失败")

    return {
        "success": True,
        "deleted_id": record_id,
        "removed_files": removed_files,
    }
