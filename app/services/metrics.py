from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import ML_DIR, Settings

MODEL_SUFFIXES = {".keras", ".h5", ".pkl", ".joblib", ".pt", ".pth"}
_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}


def _file_signature(path: Path) -> tuple[Any, ...]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def read_model_metrics(settings: Settings) -> dict[str, Any]:
    candidates = list(dict.fromkeys([
        settings.models_dir / "static_metadata.json",
        settings.models_dir / "dynamic_metadata.json",
        settings.models_dir / "evaluation.json",
        settings.models_dir / "metrics.json",
        settings.models_dir / "model_metrics.json",
        settings.models_dir / "evaluation" / "static_metrics.json",
        settings.models_dir / "evaluation" / "dynamic_metrics.json",
        ML_DIR / "evaluation" / "evaluation.json",
        ML_DIR / "evaluation" / "metrics.json",
    ]))
    metadata_paths = [*candidates, settings.models_dir]
    signature = tuple(_file_signature(path) for path in metadata_paths)
    cache_key = str(settings.models_dir)
    cached = _CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return dict(cached[1])
    metrics = None
    source = None
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        # Training metadata nests the real evaluation under test_metrics.
        if isinstance(payload, dict) and isinstance(payload.get("test_metrics"), dict):
            metrics = dict(payload["test_metrics"])
            metrics.setdefault("model_version", payload.get("model_version"))
            metrics.setdefault("recognition_type", payload.get("recognition_type"))
            source = path.name
            break
        if isinstance(payload, dict) and payload:
            metrics = payload
            source = path.name
            break
    models = sorted(
        path.name
        for path in settings.models_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ) if settings.models_dir.is_dir() else []
    version = metrics.get("model_version") if isinstance(metrics, dict) else None
    result = {
        "available": metrics is not None or bool(models),
        "model_version": version,
        "metrics": metrics,
        "source": source,
        "model_files": models,
    }
    _CACHE[cache_key] = (signature, result)
    return dict(result)
