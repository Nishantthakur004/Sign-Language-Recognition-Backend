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
from app.database.connection import (
    configure_indexes,
    ping_database,
    shutdown_database,
    startup_database,
)
from app.routes import analytics, dataset, dictionary, recognition
from app.services.dataset_manager import DatasetManager
from app.services.dictionary import DictionaryService
from app.services.predictor import PredictionService

logger = logging.getLogger(__name__)

_LOGO_URL = "https://i.ibb.co/yn3jvys4/Hand-Sign.png"

_ICONS: dict[str, str] = {
    "settings": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
    "layers": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>',
    "database": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg>',
    "cpu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>',
    "file-code": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="m10 13-2 2 2 2"/><path d="m14 17 2-2-2-2"/></svg>',
    "heart-pulse": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/></svg>',
    "play": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"/></svg>',
    "activity": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>',
    "code": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/></svg>',
    "copy": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
}

_STATUS_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; overflow-x: hidden; background: #05081c; color: #f8fafc; font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
body::before { content: ""; position: fixed; top: -22vh; right: -12vw; width: 78vw; height: 62vh; background: radial-gradient(ellipse at center, rgba(124, 58, 237, .55) 0%, rgba(88, 28, 135, .28) 42%, rgba(5, 8, 28, 0) 72%); filter: blur(30px); pointer-events: none; }
body::after { content: ""; position: fixed; bottom: -30vh; left: -18vw; width: 70vw; height: 60vh; background: radial-gradient(ellipse at center, rgba(59, 7, 100, .65) 0%, rgba(37, 99, 235, .18) 45%, rgba(5, 8, 28, 0) 72%); filter: blur(40px); pointer-events: none; }
.shell { position: relative; width: min(1600px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 56px; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 24px; }
.brand { display: flex; align-items: center; gap: 14px; color: inherit; text-decoration: none; }
.brand__logo { width: 46px; height: 46px; object-fit: contain; }
.brand__name { font-size: 27px; font-weight: 800; letter-spacing: -.02em; }
.brand__tag { padding: 5px 13px; border-radius: 999px; background: linear-gradient(135deg, #8b5cf6, #7c3aed); font-size: 14px; font-weight: 700; }
.topbar__meta { display: flex; align-items: center; gap: 14px; }
.pill { display: inline-flex; align-items: center; gap: 9px; padding: 11px 20px; border-radius: 999px; font-size: 15px; font-weight: 600; text-decoration: none; }
.pill--online { color: #4ade80; background: rgba(16, 185, 129, .10); border: 1px solid rgba(34, 197, 94, .32); }
.pill--version { color: #c7d2fe; background: rgba(79, 70, 229, .16); border: 1px solid rgba(129, 140, 248, .35); }
.dot { width: 9px; height: 9px; border-radius: 999px; background: currentColor; box-shadow: 0 0 10px currentColor; }
.hero { margin: 58px 0 40px; }
.hero h1 { margin: 0; font-size: clamp(40px, 5vw, 60px); font-weight: 800; letter-spacing: -.03em; line-height: 1.05; }
.hero h1 span { background: linear-gradient(100deg, #d8b4fe, #a855f7 55%, #8b5cf6); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { display: flex; align-items: center; gap: 14px; margin: 18px 0 0; font-size: 18px; color: #cbd5e1; }
.hero p::before { content: ""; width: 4px; height: 26px; border-radius: 999px; background: linear-gradient(#c084fc, #7c3aed); }
.status-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 18px; }
.status-card { display: flex; align-items: center; gap: 14px; padding: 20px 18px; border: 1px solid; border-radius: 16px; background: linear-gradient(140deg, var(--tint-strong), var(--tint-soft)); box-shadow: 0 0 26px var(--glow); }
.status-card--ok { --accent: #22c55e; --tint-strong: rgba(16, 185, 129, .16); --tint-soft: rgba(16, 185, 129, .03); --glow: rgba(34, 197, 94, .18); border-color: rgba(34, 197, 94, .45); }
.status-card--warn { --accent: #eab308; --tint-strong: rgba(234, 179, 8, .15); --tint-soft: rgba(234, 179, 8, .03); --glow: rgba(234, 179, 8, .16); border-color: rgba(234, 179, 8, .5); }
.status-card--bad { --accent: #fb3b64; --tint-strong: rgba(244, 63, 94, .16); --tint-soft: rgba(244, 63, 94, .03); --glow: rgba(244, 63, 94, .18); border-color: rgba(244, 63, 94, .5); }
.status-card__icon { display: grid; place-items: center; width: 34px; height: 34px; flex: none; color: var(--accent); }
.status-card__icon svg { width: 30px; height: 30px; }
.status-card__body { display: flex; flex-direction: column; gap: 7px; min-width: 0; }
.status-card__label { font-size: 16px; font-weight: 600; color: #f1f5f9; }
.status-card__state { display: inline-flex; align-items: center; gap: 8px; font-size: 14.5px; font-weight: 600; color: var(--accent); }
.panels { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.62fr); gap: 24px; margin-top: 30px; }
.panel { padding: 22px 24px 24px; border: 1px solid rgba(99, 102, 241, .38); border-radius: 20px; background: linear-gradient(160deg, rgba(23, 32, 66, .75), rgba(10, 15, 38, .85)); box-shadow: 0 0 40px rgba(59, 7, 100, .25); }
.panel__head { display: flex; align-items: center; gap: 12px; color: #a78bfa; }
.panel__head svg { width: 24px; height: 24px; }
.panel__head h2 { margin: 0; font-size: 21px; font-weight: 700; color: #fff; }
.panel--dataset .panel__head { padding-bottom: 16px; border-bottom: 1px solid rgba(148, 163, 184, .16); }
.health-box { --accent: #22c55e; display: flex; flex-direction: column; align-items: center; gap: 6px; margin-top: 18px; padding: 30px 24px; border: 1px solid rgba(34, 197, 94, .4); border-radius: 16px; background: linear-gradient(160deg, rgba(16, 185, 129, .18), rgba(13, 20, 45, .35)); text-align: center; }
.health-box--warn { --accent: #eab308; border-color: rgba(234, 179, 8, .45); background: linear-gradient(160deg, rgba(234, 179, 8, .16), rgba(13, 20, 45, .35)); }
.health-check { display: grid; place-items: center; width: 58px; height: 58px; margin-bottom: 6px; border: 2px solid var(--accent); border-radius: 999px; color: var(--accent); }
.health-check svg { width: 30px; height: 30px; }
.health-box strong { font-size: 22px; font-weight: 700; color: var(--accent); }
.health-box p { margin: 0; font-size: 14.5px; color: #cbd5e1; }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 18px; }
.stat { padding: 18px; border: 1px solid rgba(148, 163, 184, .16); border-radius: 14px; background: linear-gradient(160deg, rgba(30, 41, 59, .7), rgba(15, 23, 42, .55)); }
.stat__head { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 500; color: #e2e8f0; }
.stat__head svg { width: 20px; height: 20px; flex: none; color: #a78bfa; }
.stat strong { display: block; margin-top: 12px; font-size: 34px; font-weight: 800; letter-spacing: -.02em; }
.env-bar { display: flex; align-items: center; gap: 22px; margin-top: 30px; padding: 16px 20px; border: 1px solid rgba(99, 102, 241, .42); border-radius: 16px; background: linear-gradient(120deg, rgba(23, 32, 66, .8), rgba(10, 15, 38, .85)); }
.env-bar__label { display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 700; white-space: nowrap; }
.env-bar__label svg { width: 26px; height: 26px; color: #22d3ee; }
.env-bar__divider { width: 1px; height: 38px; background: rgba(148, 163, 184, .28); }
.env-bar__value { display: flex; align-items: center; gap: 14px; flex: 1; min-width: 0; padding: 9px 10px 9px 16px; border: 1px solid rgba(148, 163, 184, .14); border-radius: 10px; background: rgba(2, 6, 23, .55); }
.env-bar__value code { flex: 1; min-width: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 15px; color: #e2e8f0; overflow-wrap: anywhere; }
.env-bar__value code span { color: #22d3ee; }
.copy-button { display: grid; place-items: center; width: 40px; height: 40px; flex: none; border: 1px solid rgba(148, 163, 184, .22); border-radius: 9px; background: transparent; color: #cbd5e1; cursor: pointer; transition: background .15s ease, color .15s ease; }
.copy-button:hover { background: rgba(148, 163, 184, .12); color: #fff; }
.copy-button svg { width: 20px; height: 20px; }
.copy-button .icon-check { display: none; color: #4ade80; }
.copy-button.is-copied .icon-copy { display: none; }
.copy-button.is-copied .icon-check { display: block; }
@media (max-width: 1180px) { .status-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 900px) { .panels { grid-template-columns: 1fr; } .env-bar { flex-wrap: wrap; } .env-bar__divider { display: none; } .env-bar__value { flex-basis: 100%; } }
@media (max-width: 680px) { .shell { width: calc(100% - 32px); padding-top: 24px; } .status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .stats { grid-template-columns: 1fr; } .topbar { flex-wrap: wrap; } .hero { margin-top: 40px; } .brand__name { font-size: 23px; } }
"""


def _status_card(icon: str, label: str, state: str, text: str) -> str:
    return (
        f'<article class="status-card status-card--{state}">'
        f'<span class="status-card__icon">{_ICONS[icon]}</span>'
        f'<div class="status-card__body">'
        f'<span class="status-card__label">{html.escape(label)}</span>'
        f'<span class="status-card__state"><span class="dot"></span>{html.escape(text)}</span>'
        f"</div></article>"
    )


async def _status_snapshot(app: FastAPI, *, include_dataset: bool = False) -> dict[str, Any]:
    db = app.state.db
    database_reachable = await ping_database(db) if db is not None else False
    if database_reachable and not getattr(app.state, "db_indexes_ready", False):
        try:
            await configure_indexes(db)
        except PyMongoError:
            logger.warning("MongoDB is reachable but index creation failed", exc_info=True)
        else:
            app.state.db_indexes_ready = True
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
        database_ok = snapshot["database"]["available"]
        model_ok = snapshot["model"]["available"]
        dataset_ok = dataset["available"]
        manifest_ok = dataset["manifest_exists"]
        health_class = "" if overall_ok else " health-box--warn"
        health_title = "Healthy" if overall_ok else "Degraded"
        health_text = (
            "All core backend components are operational."
            if overall_ok
            else "Some backend components need attention."
        )
        cards = "".join(
            (
                _status_card("settings", "Service", "ok", "Online"),
                _status_card(
                    "layers",
                    "Overall",
                    "ok" if overall_ok else "warn",
                    "Healthy" if overall_ok else "Degraded",
                ),
                _status_card(
                    "database",
                    "Database",
                    "ok" if database_ok else "bad",
                    "Available" if database_ok else "Unavailable",
                ),
                _status_card(
                    "database",
                    "Dataset",
                    "ok" if dataset_ok else "bad",
                    "Available" if dataset_ok else "Unavailable",
                ),
                _status_card(
                    "cpu",
                    "Model",
                    "ok" if model_ok else "bad",
                    "Available" if model_ok else "Unavailable",
                ),
                _status_card(
                    "file-code",
                    "Manifest",
                    "ok" if manifest_ok else "bad",
                    "Available" if manifest_ok else "Missing",
                ),
            )
        )
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SignTrans API — Backend Status</title>
  <link rel="icon" type="image/png" href="{_LOGO_URL}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>{_STATUS_CSS}</style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <a class="brand" href="/">
        <img class="brand__logo" src="{_LOGO_URL}" alt="SignTrans logo">
        <span class="brand__name">SignTrans</span>
        <span class="brand__tag">API</span>
      </a>
      <div class="topbar__meta">
        <a class="pill pill--online" href="/health"><span class="dot"></span>Backend Online</a>
        <a class="pill pill--version" href="/docs">v1.0.0</a>
      </div>
    </header>

    <section class="hero">
      <h1>Backend <span>Status</span></h1>
      <p>Recognition, dictionary, analytics, and dataset APIs.</p>
    </section>

    <section class="status-grid">{cards}</section>

    <section class="panels">
      <article class="panel panel--health">
        <header class="panel__head">{_ICONS["heart-pulse"]}<h2>Health</h2></header>
        <div class="health-box{health_class}">
          <span class="health-check">{_ICONS["check"]}</span>
          <strong>{health_title}</strong>
          <p>{health_text}</p>
        </div>
      </article>
      <article class="panel panel--dataset">
        <header class="panel__head">{_ICONS["database"]}<h2>Dataset Information</h2></header>
        <div class="stats">
          <div class="stat"><span class="stat__head">{_ICONS["play"]}Video Clips</span><strong>{dataset["clips"]:,}</strong></div>
          <div class="stat"><span class="stat__head">{_ICONS["activity"]}Cached Landmarks</span><strong>{dataset["cached_landmarks"]:,}</strong></div>
          <div class="stat"><span class="stat__head">{_ICONS["layers"]}Classes</span><strong>{dataset["classes"]:,}</strong></div>
        </div>
      </article>
    </section>

    <section class="env-bar">
      <div class="env-bar__label">{_ICONS["code"]}Frontend Environment</div>
      <span class="env-bar__divider"></span>
      <div class="env-bar__value">
        <code id="env-value">VITE_API_BASE_URL=<span>{html.escape(base_url)}/api</span></code>
        <button class="copy-button" id="copy-env" type="button" aria-label="Copy environment value">
          <span class="icon-copy">{_ICONS["copy"]}</span>
          <span class="icon-check">{_ICONS["check"]}</span>
        </button>
      </div>
    </section>
  </main>
  <script>
    const copyButton = document.getElementById('copy-env');
    const envValue = document.getElementById('env-value');
    copyButton.addEventListener('click', async () => {{
      try {{
        await navigator.clipboard.writeText(envValue.textContent.trim());
        copyButton.classList.add('is-copied');
        setTimeout(() => copyButton.classList.remove('is-copied'), 1600);
      }} catch (error) {{
        /* Clipboard access is unavailable in this context. */
      }}
    }});
  </script>
</body>
</html>"""
        return HTMLResponse(content=page)

    app.include_router(recognition.router)
    app.include_router(dictionary.router)
    app.include_router(analytics.router)
    app.include_router(dataset.router)
    return app


app = create_app()
