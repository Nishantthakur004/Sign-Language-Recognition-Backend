from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mongomock_motor import AsyncMongoMockClient
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402


class FakePredictionService:
    """Deterministic stand-in for the real model adapter; never used in production."""

    available = True

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, dict]] = {}
        self._counter = 0

    async def predict(self, image_bytes: bytes, *, landmarks=None, sequence=None) -> dict:
        return {
            "sign": "A",
            "confidence": 0.93,
            "category": "alphabet",
            "model_version": "test-static",
            "recognition_type": "static",
            "top_k": [
                {"rank": 1, "sign": "A", "confidence": 0.93, "category": "alphabet"},
                {"rank": 2, "sign": "B", "confidence": 0.04, "category": "alphabet"},
            ],
            "hand_detected": True,
            "landmarks": None,
        }

    def cache_prediction(self, user_id, prediction: dict) -> dict:
        self._counter += 1
        prediction_id = f"prediction-{self._counter}"
        self._cache[prediction_id] = (str(user_id), dict(prediction))
        return {**prediction, "prediction_id": prediction_id}

    def cached_prediction(self, prediction_id: str, user_id) -> dict | None:
        entry = self._cache.get(prediction_id)
        if entry is None or entry[0] != str(user_id):
            return None
        return dict(entry[1])


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def client():
    app = create_app(get_settings())
    db = AsyncMongoMockClient()["signtrans_test"]
    asyncio.run(
        db.recognition_history.create_index(
            [("user_id", 1), ("prediction_id", 1)],
            unique=True,
            partialFilterExpression={"prediction_id": {"$type": "string"}},
        )
    )
    app.state.db = db
    app.state.db_available = True
    app.state.db_indexes_ready = True
    app.state.prediction_service = FakePredictionService()
    return TestClient(app)


def test_prediction_history_analytics_and_export(client):
    prediction = client.post(
        "/api/recognition/predict",
        files={"file": ("frame.jpg", image_bytes(), "image/jpeg")},
    )
    assert prediction.status_code == 200, prediction.text
    body = prediction.json()
    assert body["sign"] == "A"
    assert body["prediction_id"]

    saved = client.post(
        "/api/recognition/history",
        json={"prediction_id": body["prediction_id"], "session_id": "session-1", "duration_ms": 1200},
    )
    assert saved.status_code == 201, saved.text
    history_id = saved.json()["id"]

    duplicate = client.post(
        "/api/recognition/history",
        json={"prediction_id": body["prediction_id"]},
    )
    assert duplicate.status_code == 409

    listing = client.get("/api/recognition/history")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["sign"] == "A"

    overview = client.get("/api/analytics/overview")
    assert overview.status_code == 200
    assert overview.json()["total_recognitions"] == 1
    assert overview.json()["total_sessions"] == 1

    export = client.get("/api/recognition/history/export")
    assert export.status_code == 200
    assert "sign" in export.text and "A" in export.text

    deleted = client.delete(f"/api/recognition/history/{history_id}")
    assert deleted.status_code == 200
    assert client.get("/api/recognition/history").json()["total"] == 0


def test_history_rejects_unknown_prediction_id(client):
    response = client.post(
        "/api/recognition/history",
        json={"prediction_id": "not-a-real-prediction"},
    )
    assert response.status_code == 404


def test_dictionary_endpoints_are_public_and_consistent(client):
    listing = client.get("/api/dictionary")
    assert listing.status_code == 200
    payload = listing.json()
    # The dictionary is populated from the real dataset report when present.
    assert payload["total"] == len(payload["items"]) or payload["total"] >= len(payload["items"])
    assert client.get("/api/dictionary/categories").status_code == 200
    assert client.get("/api/dictionary/does-not-exist").status_code == 404
