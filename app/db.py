from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data"
DB_PATH = DB_DIR / "app.db"


def get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                annotated_path TEXT,
                predicted_class TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_mode TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_prediction(payload: dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                filename, original_path, annotated_path, predicted_class,
                predicted_label, confidence, model_mode, raw_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["filename"],
                payload["original_path"],
                payload.get("annotated_path"),
                payload["predicted_class"],
                payload["predicted_label"],
                payload["confidence"],
                payload["model_mode"],
                payload["raw_json"],
                payload["created_at"],
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_predictions(limit: int = 20) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def stats() -> dict[str, Any]:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        by_class_rows = conn.execute(
            """
            SELECT predicted_label, COUNT(*) AS count
            FROM predictions
            GROUP BY predicted_label
            ORDER BY count DESC, predicted_label ASC
            """
        ).fetchall()
        latest = conn.execute(
            "SELECT created_at FROM predictions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return {
        "total_predictions": total,
        "by_class": [dict(row) for row in by_class_rows],
        "latest_prediction_at": latest[0] if latest else None,
    }
