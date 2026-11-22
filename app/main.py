from __future__ import annotations

import asyncio
import html
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pymongo.errors import PyMongoError

from app.config import Settings, get_settings
from app.database.connection import ping_database, shutdown_database, startup_database
from app.routes import analytics, dataset, dictionary, recognition
from app.services.dataset_manager import DatasetManager
from app.services.dictionary import DictionaryService
from app.services.predictor import PredictionService

logger = logging.getLogger(__name__)

_STATUS_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; background: radial-gradient(circle at top, #1e293b, #0b1120 60%); color: #e2e8f0; }
main { width: min(760px, calc(100% - 40px)); margin: 0 auto; padding: 64px 0; }
.eyebrow { margin: 0 0 8px; color: #38bdf8; font-size: 12px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }
h1 { margin: 0; font-size: clamp(32px, 6vw, 48px); letter-spacing: -.03em; }
.subtitle { margin: 12px 0 32px; color: #94a3b8; line-height: 1.6; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.card { padding: 18px; border: 1px solid #1e293b; border-radius: 14px; background: rgba(15, 23, 42, .72); }
.label { display: block; margin-bottom: 8px; color: #94a3b8; font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; }
.badge::before { content: ""; width: 8px; height: 8px; border-radius: 999px; background: currentColor; }
.ok { color: #4ade80; }
.warn { color: #fbbf24; }
.bad { color: #f87171; }
.details { margin-top: 24px; border: 1px solid #1e293b; border-radius: 14px; overflow: hidden; }
.row { display: flex; justify-content: space-between; gap: 16px; padding: 14px 18px; border-bottom: 1px solid #1e293b; }
.row:last-child { border-bottom: 0; }
.row span:first-child { color: #94a3b8; }
.row span:last-child { text-align: right; overflow-wrap: anywhere; }
.env { margin-top: 24px; padding: 18px; border: 1px solid #1e293b; border-radius: 14px; background: rgba(15, 23, 42, .72); }
.env code { display: block; margin-top: 8px; color: #7dd3fc; overflow-wrap: anywhere; }
nav { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 28px; }
a { color: #7dd3fc; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def _badge(available: bool, ok: str = "Available", bad: str = "Unavailable") -> str:
    return f'<span class="badge {"ok" if available else "bad"}">{ok if available else bad}</span>'


async def _status_snapshot(app: FastAPI, *, include_dataset: bool = False) -> dict[str, Any]:
    db = app.state.db
    database_reachable = await ping_database(db) if db is not None else False
    indexes_ready = bool(getattr(app.state, "db_indexes_ready", False))
    database_available = database_reachable and indexes_ready
    app.state.db_available = database_available
    model_available = app.state.prediction_service.available
    payload: dict[str, Any] = {
        "status": "ok" if database_available and model_available else "degraded",
        "database": {"available": database_available, "indexes_ready": indexes_ready},
        "model": {"available": model_available},
    }
    if include_dataset:
        try:
            dataset = await asyncio.to_thread(app.state.dataset_manager.dataset_status)
        except Exception:
            logger.exception("Unable to read dataset status")
            dataset = {}
        dataset_available = bool(dataset.get("root_exists") and dataset.get("clips"))
        payload["dataset"] = {
            "available": dataset_available,
            "root": dataset.get("root"),
            "root_exists": bool(dataset.get("root_exists")),
            "clips": int(dataset.get("clips") or 0),
            "cached_landmarks": int(dataset.get("cached_landmarks") or 0),
            "manifest_exists": bool(dataset.get("manifest_exists")),
            "classes": int(dataset.get("class_count") or 0),
        }
        if not dataset_available:
            payload["status"] = "degraded"
    return payload


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await startup_database(app, settings)
        await asyncio.to_thread(app.state.prediction_service.load)
        try:
            yield
        finally:
            await shutdown_database(app)

    app = FastAPI(title="SignTrans API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.db = None
    app.state.db_available = False
    app.state.db_indexes_ready = False
    app.state.prediction_service = PredictionService(settings)
    app.state.dictionary_service = DictionaryService(settings)
    app.state.dataset_manager = DatasetManager(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    @app.exception_handler(PyMongoError)
    async def database_error(_: Request, exc: PyMongoError) -> JSONResponse:
        logger.error("Database operation failed: %s", exc.__class__.__name__)
        return JSONResponse(status_code=503, content={"detail": "Database is unavailable"})

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, Any]:
        return await _status_snapshot(app)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        snapshot = await _status_snapshot(app, include_dataset=True)
        base_url = str(request.base_url).rstrip("/")
        dataset = snapshot["dataset"]
        overall_ok = snapshot["status"] == "ok"
        overall_class = "ok" if overall_ok else "warn"
        overall_label = "Operational" if overall_ok else "Degraded"
        db_badge = _badge(snapshot["database"]["available"])
        model_badge = _badge(snapshot["model"]["available"])
        dataset_badge = _badge(dataset["available"])
        manifest_badge = _badge(dataset["manifest_exists"], "Available", "Missing")
        rows = (
            ("Dataset root", dataset["root"] or "not configured"),
            ("Video clips", f'{dataset["clips"]:,}'),
            ("Cached landmarks", f'{dataset["cached_landmarks"]:,}'),
            ("Classes", f'{dataset["classes"]:,}'),
        )
        details = "".join(
            f'<div class="row"><span>{html.escape(label)}</span><span>{html.escape(value)}</span></div>'
            for label, value in rows
        )
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SignTrans API status</title>
  <style>{_STATUS_CSS}</style>
</head>
<body>
  <main>
    <p class="eyebrow">SignTrans API</p>
    <h1>Backend status</h1>
    <p class="subtitle">Recognition, dictionary, analytics, and dataset APIs. The React frontend runs separately.</p>
    <section class="grid">
      <article class="card"><span class="label">Service</span><span class="badge ok">Online</span></article>
      <article class="card"><span class="label">Overall</span><span class="badge {overall_class}">{overall_label}</span></article>
      <article class="card"><span class="label">Database</span>{db_badge}</article>
      <article class="card"><span class="label">Model</span>{model_badge}</article>
      <article class="card"><span class="label">Dataset</span>{dataset_badge}</article>
      <article class="card"><span class="label">Manifest</span>{manifest_badge}</article>
    </section>
    <section class="details">{details}</section>
    <section class="env">
      <span class="label">Frontend .env</span>
      <code>VITE_API_BASE_URL={html.escape(base_url)}/api</code>
    </section>
    <nav>
      <a href="/docs">API docs</a>
      <a href="/health">Health JSON</a>
      <a href="/api/dataset/status">Dataset JSON</a>
    </nav>
  </main>
</body>
</html>"""
        return HTMLResponse(content=page)

    app.include_router(recognition.router)
    app.include_router(dictionary.router)
    app.include_router(analytics.router)
    app.include_router(dataset.router)
    return app


app = create_app()
