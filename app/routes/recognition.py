from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
import tempfile
import time
from collections import deque
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError

from app.dependencies import get_db, get_optional_user
from app.models.history import history_document, public_history
from app.models.user import object_id
from app.schemas.recognition import (
    HistoryListResponse,
    HistoryResponse,
    HistorySaveRequest,
    PredictionResult,
    normalize_landmark_data,
    parse_landmark_json,
)
from app.services.images import ImageValidationError, decode_data_url, validate_image_bytes
from app.services.predictor import ModelUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recognition"])


async def run_prediction(
    request: Request,
    image_bytes: bytes,
    user: dict[str, Any],
    *,
    landmarks: Any = None,
    sequence: Any = None,
) -> dict[str, Any]:
    service = request.app.state.prediction_service
    try:
        prediction = await service.predict(image_bytes, landmarks=landmarks, sequence=sequence)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Sign recognition model is unavailable") from exc
    except ValueError as exc:
        logger.error("Recognition model returned an invalid result: %s", exc)
        raise HTTPException(status_code=502, detail="Recognition model returned an invalid result") from exc
    except Exception as exc:
        logger.exception("Recognition model inference failed")
        raise HTTPException(status_code=502, detail="Recognition model inference failed") from exc
    return service.cache_prediction(user["_id"], prediction)


@router.post("/api/recognition/predict", response_model=PredictionResult)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    landmarks: str | None = Form(default=None),
    sequence: str | None = Form(default=None),
    user: dict[str, Any] = Depends(get_optional_user),
) -> dict[str, Any]:
    max_bytes = request.app.state.settings.max_upload_bytes
    try:
        normalized_landmarks = parse_landmark_json(landmarks)
        normalized_sequence = parse_landmark_json(sequence, sequence=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        image_bytes = await file.read(max_bytes + 1)
        validate_image_bytes(image_bytes, max_bytes, file.content_type)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        await file.close()
    return await run_prediction(
        request,
        image_bytes,
        user,
        landmarks=normalized_landmarks,
        sequence=normalized_sequence,
    )


@router.post("/api/recognition/predict-sequence", response_model=PredictionResult)
async def predict_sequence(
    request: Request,
    payload: dict[str, Any] = Body(...),
    user: dict[str, Any] = Depends(get_optional_user),
) -> dict[str, Any]:
    """Predict a word from browser MediaPipe landmark frames (no image upload)."""
    sequence = payload.get("sequence")
    if not isinstance(sequence, list) or not sequence:
        raise HTTPException(status_code=400, detail="sequence must be a non-empty list of landmark frames")
    return await run_prediction(request, b"", user, sequence=sequence)


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@router.post("/api/recognition/recognize-video")
async def recognize_video(
    request: Request,
    file: UploadFile = File(...),
    user: dict[str, Any] = Depends(get_optional_user),
) -> dict[str, Any]:
    """Recognize a sequence of ISL words from an uploaded sign video."""
    settings = request.app.state.settings
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Upload an MP4, MOV, AVI, MKV, WEBM, or M4V video",
        )
    limit = settings.max_video_upload_bytes
    try:
        data = await file.read(limit + 1)
    finally:
        await file.close()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded video is empty")
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds the {settings.max_video_upload_mb} MB limit",
        )

    from ml.video_inference import (
        VideoRecognitionError,
        build_windows,
        extract_video_frames,
        summarize_predictions,
    )

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(data)
        handle.close()
        extracted = await asyncio.to_thread(extract_video_frames, handle.name)
        windows, spans = build_windows(extracted["frames"], extracted["fps"])
        predictions = await request.app.state.prediction_service.predict_sequences(windows)
        return summarize_predictions(
            predictions,
            spans,
            fps=extracted["fps"],
            total_frames=extracted["sampled_frames"],
            detected_frames=extracted["detected"],
        )
    except VideoRecognitionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Sign recognition model is unavailable") from exc
    except ValueError as exc:
        logger.error("Video recognition returned an invalid result: %s", exc)
        raise HTTPException(status_code=502, detail="Recognition model returned an invalid result") from exc
    except Exception as exc:
        logger.exception("Video recognition failed")
        raise HTTPException(status_code=502, detail="Video recognition failed") from exc
    finally:
        Path(handle.name).unlink(missing_ok=True)


@router.post("/api/recognition/history", response_model=HistoryResponse, status_code=201)
async def save_history(
    payload: HistorySaveRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, Any]:
    prediction = request.app.state.prediction_service.cached_prediction(
        payload.prediction_id,
        user["_id"],
    )
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found or expired")
    if payload.thumbnail is not None:
        try:
            decode_data_url(payload.thumbnail, request.app.state.settings.max_upload_bytes)
        except ImageValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=f"Invalid thumbnail: {exc}") from exc
    document = history_document(
        user["_id"],
        payload,
        prediction,
        prediction_id=payload.prediction_id,
        thumbnail=payload.thumbnail,
    )
    try:
        result = await db.recognition_history.insert_one(document)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="Prediction has already been saved") from exc
    document["_id"] = result.inserted_id
    return public_history(document)


@router.get("/api/recognition/history", response_model=HistoryListResponse)
async def list_history(
    search: str | None = Query(default=None, max_length=100),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    history_date: date | None = Query(default=None, alias="date"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, Any]:
    filters: dict[str, Any] = {"user_id": user["_id"]}
    if search and search.strip():
        pattern = re.escape(search.strip())
        filters["$or"] = [
            {"sign": {"$regex": pattern, "$options": "i"}},
            {"category": {"$regex": pattern, "$options": "i"}},
        ]
    created: dict[str, datetime] = {}
    if start_date is not None:
        created["$gte"] = start_date
    if end_date is not None:
        created["$lte"] = end_date
    if history_date is not None:
        start = datetime.combine(history_date, datetime_time.min)
        created = {"$gte": start, "$lt": start + timedelta(days=1)}
    if created:
        filters["created_at"] = created
    total = await db.recognition_history.count_documents(filters)
    cursor = (
        db.recognition_history.find(filters, {"thumbnail": 0})
        .sort("created_at", DESCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    items = await cursor.to_list(length=page_size)
    return {
        "items": [public_history(item, include_thumbnail=False) for item in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/api/recognition/history/export")
async def export_history(
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> Response:
    rows = await db.recognition_history.find(
        {"user_id": user["_id"]},
        {"thumbnail": 0},
    ).sort("created_at", DESCENDING).to_list(length=None)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    fields = [
        "id",
        "prediction_id",
        "sign",
        "confidence",
        "category",
        "model_version",
        "recognition_type",
        "session_id",
        "duration_ms",
        "created_at",
    ]
    writer.writerow(fields)
    for row in rows:
        values = []
        for field in fields:
            value = str(row["_id"]) if field == "id" else row.get(field, "")
            if isinstance(value, datetime):
                value = value.isoformat()
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                value = "'" + value
            values.append("" if value is None else value)
        writer.writerow(values)
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="recognition-history.csv"'},
    )


@router.delete("/api/recognition/history")
async def clear_history(
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, int]:
    result = await db.recognition_history.delete_many({"user_id": user["_id"]})
    return {"deleted_count": result.deleted_count}


@router.get("/api/recognition/history/{history_id}", response_model=HistoryResponse)
async def history_detail(
    history_id: str,
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, Any]:
    identifier = object_id(history_id)
    if identifier is None:
        raise HTTPException(status_code=400, detail="Invalid history id")
    item = await db.recognition_history.find_one({"_id": identifier, "user_id": user["_id"]})
    if item is None:
        raise HTTPException(status_code=404, detail="History item not found")
    return public_history(item)


@router.delete("/api/recognition/history/{history_id}")
async def delete_history(
    history_id: str,
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, str]:
    identifier = object_id(history_id)
    if identifier is None:
        raise HTTPException(status_code=400, detail="Invalid history id")
    result = await db.recognition_history.delete_one({"_id": identifier, "user_id": user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="History item not found")
    return {"message": "History item deleted"}


def websocket_error(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"code": code, "message": message}}


@router.websocket("/ws/recognition")
async def recognition_websocket(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings

    await websocket.accept()
    timestamps: deque[float] = deque()
    while True:
        try:
            payload = await websocket.receive_json()
        except WebSocketDisconnect:
            break
        except (json.JSONDecodeError, ValueError):
            await websocket.send_json(
                websocket_error("invalid_json", "Send a JSON object with a landmark sequence")
            )
            continue

        now = time.monotonic()
        while timestamps and now - timestamps[0] >= settings.ws_rate_window_seconds:
            timestamps.popleft()
        if len(timestamps) >= settings.ws_rate_limit:
            await websocket.send_json(websocket_error("rate_limited", "Frame rate limit exceeded"))
            continue
        timestamps.append(now)

        if not isinstance(payload, dict):
            await websocket.send_json(
                websocket_error("invalid_frame", "Expected {sequence: [[...]]} or {frame: dataURL}")
            )
            continue
        try:
            if isinstance(payload.get("sequence"), list):
                prediction = await websocket.app.state.prediction_service.predict(
                    b"", sequence=payload["sequence"]
                )
            elif isinstance(payload.get("frame"), str):
                image_bytes = decode_data_url(payload["frame"], settings.max_upload_bytes)
                prediction = await websocket.app.state.prediction_service.predict(image_bytes)
            else:
                await websocket.send_json(
                    websocket_error("invalid_frame", "Expected {sequence: [[...]]} or {frame: dataURL}")
                )
                continue
        except ImageValidationError as exc:
            await websocket.send_json(websocket_error(exc.code, str(exc)))
        except ModelUnavailableError:
            await websocket.send_json(websocket_error("model_unavailable", "Sign recognition model is unavailable"))
        except ValueError:
            await websocket.send_json(websocket_error("invalid_model_result", "Recognition model returned an invalid result"))
        except Exception:
            logger.exception("WebSocket recognition inference failed")
            await websocket.send_json(websocket_error("inference_failed", "Recognition model inference failed"))
        else:
            await websocket.send_json({"type": "prediction", "prediction": prediction})
