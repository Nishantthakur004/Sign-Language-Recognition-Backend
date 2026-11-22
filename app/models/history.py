from __future__ import annotations

from datetime import datetime
from typing import Any


def public_history(item: dict[str, Any], include_thumbnail: bool = True) -> dict[str, Any]:
    return {
        "id": str(item["_id"]),
        "sign": item["sign"],
        "confidence": item["confidence"],
        "category": item.get("category"),
        "model_version": item.get("model_version"),
        "recognition_type": item.get("recognition_type"),
        "prediction_id": item.get("prediction_id"),
        "session_id": item.get("session_id"),
        "duration_ms": item.get("duration_ms", 0),
        "thumbnail": item.get("thumbnail") if include_thumbnail else None,
        "created_at": item.get("created_at"),
    }


def history_document(
    user_id: Any,
    payload: Any,
    prediction: dict[str, Any],
    *,
    prediction_id: str,
    thumbnail: str | None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "prediction_id": prediction_id,
        "sign": prediction["sign"],
        "confidence": prediction["confidence"],
        "category": prediction.get("category"),
        "model_version": prediction.get("model_version"),
        "recognition_type": prediction.get("recognition_type"),
        "session_id": payload.session_id,
        "duration_ms": payload.duration_ms,
        "thumbnail": thumbnail,
        "created_at": datetime.utcnow(),
    }
