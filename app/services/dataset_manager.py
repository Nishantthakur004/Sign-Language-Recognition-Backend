"""Dataset and model management backed by the local ISL40 word dataset.

The ISL40 release ships MP4 clips plus a JSON provenance sidecar per clip.
Landmarks are extracted once with MediaPipe and cached as ``.h5``; training runs
as a separate ``python backend/ml/isl40.py train`` process so the FastAPI server
never blocks.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from app.config import BACKEND_DIR, ML_DIR, Settings

logger = logging.getLogger(__name__)


def _ml() -> None:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))


class DatasetManager:
    def __init__(self, settings: Settings):
        _ml()
        self.settings = settings
        self._lock = threading.Lock()
        self._prepare_thread: threading.Thread | None = None
        self._train_process: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # ISL40 dataset status
    # ------------------------------------------------------------------
    def dataset_status(self) -> dict[str, Any]:
        root = self.settings.isl40_root
        manifest_path = self.settings.isl40_manifest
        manifest = self._read_json(manifest_path)
        clip_count = 0
        if root.is_dir():
            clip_count = sum(1 for _ in root.rglob("*.mp4"))
        cached = 0
        if self.settings.isl40_cache.is_dir():
            cached = sum(1 for _ in self.settings.isl40_cache.rglob("*.h5"))
        raw_classes = manifest.get("classes", [])
        class_count = _count(raw_classes) or _count(manifest.get("samples"))
        # Deployments may ship the manifest without the large MP4 corpus, so fall
        # back to manifest totals when the clip files are not present on disk.
        manifest_clips = _count(manifest.get("total_sequences")) or _count(manifest.get("records"))
        manifest_cached = _count(manifest.get("records"))
        payload: dict[str, Any] = {
            "name": "ISL40 (vidit031/isl-isolated-40words)",
            "source": manifest.get("source"),
            "root": str(root),
            "root_exists": root.is_dir(),
            "clips": clip_count or manifest_clips,
            "cached_landmarks": cached or manifest_cached,
            "manifest": str(manifest_path),
            "manifest_exists": manifest_path.is_file(),
            "signers": manifest.get("users", []),
            "classes": raw_classes if isinstance(raw_classes, list) else [],
            "class_count": class_count,
            "total_sequences": manifest.get("total_sequences") or manifest_clips,
            "split_counts": manifest.get("split_counts", {}),
            "class_distribution": manifest.get("class_distribution", {}),
        }
        if manifest:
            payload.update(
                {
                    "prepared_at": manifest.get("created_at"),
                    "sequence_length": manifest.get("sequence_length"),
                    "feature_dim": manifest.get("feature_dim"),
                }
            )
        return payload

    # ------------------------------------------------------------------
    # Landmark preparation (background)
    # ------------------------------------------------------------------
    def start_prepare(self) -> dict[str, Any]:
        with self._lock:
            if self._prepare_thread and self._prepare_thread.is_alive():
                raise _Busy("Landmark preparation is already running")
            if not self.settings.isl40_root.is_dir():
                raise _DatasetUnavailable("The ISL40 dataset directory is missing")
            status_file = self.settings.prepare_status_file

            def runner() -> None:
                self._write_json(
                    status_file,
                    {"state": "preparing", "updated_at": _now(), "detail": "Extracting landmarks"},
                )
                log_file = self.settings.training_log_file
                log_file.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(ML_DIR / "isl40.py"),
                    "manifest",
                    "--force",
                ]
                try:
                    with log_file.open("ab") as log_handle:
                        result = subprocess.run(
                            command,
                            cwd=str(BACKEND_DIR),
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            env={**os.environ},
                        )
                    if result.returncode == 0:
                        self._write_json(
                            status_file,
                            {"state": "completed", "updated_at": _now()},
                        )
                    else:
                        self._write_json(
                            status_file,
                            {"state": "failed", "updated_at": _now(), "error": f"exit code {result.returncode}"},
                        )
                except Exception as exc:  # noqa: BLE001
                    self._write_json(status_file, {"state": "failed", "updated_at": _now(), "error": str(exc)})

            self._prepare_thread = threading.Thread(target=runner, daemon=True)
            self._prepare_thread.start()
        return {"state": "started"}

    def prepare_status(self) -> dict[str, Any]:
        payload = self._read_json(self.settings.prepare_status_file)
        if payload:
            payload["active"] = bool(self._prepare_thread and self._prepare_thread.is_alive())
        return payload or {"state": "idle", "active": False}

    # ------------------------------------------------------------------
    # Model training (separate process, never blocks the API)
    # ------------------------------------------------------------------
    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._train_process and self._train_process.poll() is None:
                raise _Busy("A training run is already active")
            manifest_path = self.settings.isl40_manifest
            if not manifest_path.is_file():
                raise _DatasetUnavailable("The ISL40 manifest is missing; prepare the dataset first")

            status_file = self.settings.training_status_file
            log_file = self.settings.training_log_file
            log_file.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(ML_DIR / "train.py"),
                "words",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(self.settings.models_dir),
                "--architecture",
                str(payload.get("architecture") or "bilstm"),
                "--epochs",
                str(int(payload.get("epochs") or 30)),
                "--batch-size",
                str(int(payload.get("batch_size") or 32)),
                "--learning-rate",
                str(float(payload.get("learning_rate") or 1e-3)),
                "--max-per-class",
                str(int(payload.get("max_per_class") or 0) if payload.get("max_per_class") else 0),
                "--augment-copies",
                str(int(payload.get("augment_copies") if payload.get("augment_copies") is not None else 2)),
                "--status-file",
                str(status_file),
            ]
            self._write_json(
                status_file,
                {
                    "state": "preparing",
                    "updated_at": _now(),
                    "architecture": payload.get("architecture") or "bilstm",
                    "manifest": str(manifest_path),
                },
            )
            log_handle = log_file.open("ab")
            self._train_process = subprocess.Popen(
                command,
                cwd=str(BACKEND_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env={**os.environ},
            )
        return {"state": "started", "pid": self._train_process.pid}

    def training_status(self) -> dict[str, Any]:
        payload = self._read_json(self.settings.training_status_file) or {}
        process = self._train_process
        if process and process.poll() is not None and payload.get("state") in {"preparing", "training"}:
            payload["state"] = "failed"
            payload["error"] = "Training process exited before writing a completed status"
        payload["active"] = bool(process and process.poll() is None)
        if process and process.poll() is not None:
            payload["exit_code"] = process.poll()
        return payload

    def stop_training(self) -> dict[str, str]:
        process = self._train_process
        if process and process.poll() is None:
            process.terminate()
            return {"state": "stopping"}
        return {"state": "idle"}

    def model_metrics(self) -> dict[str, Any]:
        metadata = self._read_json(self.settings.models_dir / "dynamic_metadata.json")
        if not metadata:
            return {"available": False}
        payload = dict(metadata)
        payload["available"] = True
        payload["class_count"] = len(metadata.get("class_names") or [])
        payload.pop("class_names", None)
        evaluation_dir = self.settings.models_dir / "evaluation"
        payload["evaluation_files"] = sorted(
            path.name for path in evaluation_dir.glob("dynamic_*") if path.is_file()
        )
        return payload

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_status(),
            "model": self.model_metrics(),
            "training": self.training_status(),
            "prepare": self.prepare_status(),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class _Busy(Exception):
    pass


class _DatasetUnavailable(Exception):
    pass


def _now() -> str:
    from ml.models.artifacts import utc_now

    return utc_now()


def _count(value: Any) -> int:
    """Coerce manifest counts, which may be ints or lists of records."""
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
