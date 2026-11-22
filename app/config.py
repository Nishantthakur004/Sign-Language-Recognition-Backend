from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
ML_DIR = BACKEND_DIR / "ml"

# Hosted frontends (Vercel / GitHub Pages) are allowed by default so the
# deployed UI can reach the API without extra configuration. Override with
# CORS_ORIGIN_REGEX to restrict this to a specific origin.
DEFAULT_CORS_ORIGIN_REGEX = r"https://([a-z0-9-]+\.)*(vercel\.app|github\.io)"

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    # The backend owns its configuration; no frontend or repo-root env is read.
    load_dotenv(BACKEND_DIR / ".env", override=False)


def _csv_env(name: str, default: str = "") -> list[str]:
    return [value.strip() for value in os.getenv(name, default).split(",") if value.strip()]


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_db: str
    cors_origins: list[str]
    cors_origin_regex: str
    max_upload_mb: int
    max_video_upload_mb: int
    environment: str
    models_dir: Path
    dataset_report: Path
    dataset_root: Path
    ws_rate_limit: int
    ws_rate_window_seconds: float
    prediction_ttl_seconds: float = 300.0
    prediction_cache_size: int = 1000
    ws_max_message_bytes: int = 8 * 1024 * 1024
    max_image_width: int = 4096
    max_image_height: int = 4096
    max_image_pixels: int = 16_000_000
    thumbnail_max_bytes: int = 512 * 1024
    thumbnail_max_width: int = 1280
    thumbnail_max_height: int = 1280
    thumbnail_max_pixels: int = 1_000_000
    isl40_root: Path = ML_DIR / "datasets" / "ISL40"
    isl40_cache: Path = ML_DIR / "datasets" / "ISL40" / ".cache" / "landmarks"
    isl40_manifest: Path = ML_DIR / "models" / "isl40_manifest.json"
    training_status_file: Path = ML_DIR / "models" / "training_status.json"
    prepare_status_file: Path = ML_DIR / "models" / "prepare_status.json"
    training_log_file: Path = ML_DIR / "models" / "training.log"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_video_upload_bytes(self) -> int:
        return self.max_video_upload_mb * 1024 * 1024


def _env_path(name: str, default: Path) -> Path:
    """Resolve environment path settings; relative values are taken from backend/."""
    path = Path(os.getenv(name, str(default)))
    return path if path.is_absolute() else (BACKEND_DIR / path)


@lru_cache
def get_settings() -> Settings:
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    default_origins = (
        "http://localhost:3000,http://localhost:5173" if environment != "production" else ""
    )
    origins = _csv_env("CORS_ORIGINS", default_origins)
    origin_regex = os.getenv("CORS_ORIGIN_REGEX", "").strip() or DEFAULT_CORS_ORIGIN_REGEX
    dataset_root = _env_path("DATASET_ROOT", ML_DIR / "datasets" / "processed").resolve()
    models_dir = _env_path("MODELS_DIR", ML_DIR / "models").resolve()
    settings = Settings(
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_db=os.getenv("MONGODB_DB", "signtrans"),
        cors_origins=origins,
        cors_origin_regex=origin_regex,
        max_upload_mb=max(1, int(os.getenv("MAX_UPLOAD_MB", "5"))),
        max_video_upload_mb=max(1, int(os.getenv("MAX_VIDEO_UPLOAD_MB", "100"))),
        environment=environment,
        models_dir=models_dir,
        dataset_report=_env_path("DATASET_REPORT", models_dir / "dataset_report.json").resolve(),
        dataset_root=dataset_root,
        ws_rate_limit=max(1, int(os.getenv("WS_RATE_LIMIT", "12"))),
        ws_rate_window_seconds=max(0.1, float(os.getenv("WS_RATE_WINDOW_SECONDS", "1"))),
        prediction_ttl_seconds=max(1.0, float(os.getenv("PREDICTION_TTL_SECONDS", "300"))),
        prediction_cache_size=max(1, int(os.getenv("PREDICTION_CACHE_SIZE", "1000"))),
        ws_max_message_bytes=max(1024, int(os.getenv("WS_MAX_MESSAGE_BYTES", str(8 * 1024 * 1024)))),
        isl40_root=_env_path("ISL40_ROOT", ML_DIR / "datasets" / "ISL40").resolve(),
        isl40_cache=_env_path("ISL40_CACHE", ML_DIR / "datasets" / "ISL40" / ".cache" / "landmarks").resolve(),
        isl40_manifest=_env_path("ISL40_MANIFEST", models_dir / "isl40_manifest.json").resolve(),
        training_status_file=_env_path("TRAINING_STATUS_FILE", models_dir / "training_status.json").resolve(),
        prepare_status_file=_env_path("PREPARE_STATUS_FILE", models_dir / "prepare_status.json").resolve(),
        training_log_file=_env_path("TRAINING_LOG_FILE", models_dir / "training.log").resolve(),
    )
    if "*" in settings.cors_origins:
        raise RuntimeError("CORS_ORIGINS must be an explicit origin whitelist, not '*'")
    if settings.cors_origin_regex in {"*", ".*", "^.*$"}:
        raise RuntimeError("CORS_ORIGIN_REGEX must not allow every origin")
    if not settings.cors_origins and not settings.cors_origin_regex:
        raise RuntimeError(
            "CORS_ORIGINS must list the frontend origin(s) allowed to call this backend"
        )
    return settings
