"""ISL40 dataset and model management endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.dataset_manager import _Busy, _DatasetUnavailable

# This build intentionally has no login gate, so dataset/model management is
# reachable without a session. Put these routes behind admin auth before
# exposing the API publicly.
router = APIRouter(prefix="/api", tags=["dataset"])


class TrainRequest(BaseModel):
    architecture: str = Field(default="bilstm", pattern="^(lstm|gru|bilstm)$")
    epochs: int = Field(default=30, ge=1, le=200)
    batch_size: int = Field(default=32, ge=4, le=256)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    max_per_class: int = Field(default=0, ge=0, le=500)
    augment_copies: int = Field(default=2, ge=0, le=5)


def _manager(request: Request) -> Any:
    manager = getattr(request.app.state, "dataset_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Dataset manager is unavailable")
    return manager


def _errors(exc: Exception) -> HTTPException:
    if isinstance(exc, _Busy):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


@router.get("/dataset/status")
async def dataset_status(request: Request) -> dict[str, Any]:
    manager = _manager(request)
    return await asyncio.to_thread(manager.dataset_status)


@router.post("/dataset/prepare", status_code=202)
async def dataset_prepare(request: Request) -> dict[str, Any]:
    manager = _manager(request)
    try:
        return await asyncio.to_thread(manager.start_prepare)
    except (_Busy, _DatasetUnavailable) as exc:
        raise _errors(exc) from exc


@router.get("/dataset/prepare/status")
async def dataset_prepare_status(request: Request) -> dict[str, Any]:
    manager = _manager(request)
    return await asyncio.to_thread(manager.prepare_status)


@router.post("/model/train", status_code=202)
async def model_train(payload: TrainRequest, request: Request) -> dict[str, Any]:
    manager = _manager(request)
    try:
        return await asyncio.to_thread(manager.start_training, payload.model_dump())
    except (_Busy, _DatasetUnavailable) as exc:
        raise _errors(exc) from exc


@router.post("/model/stop")
async def model_stop(request: Request) -> dict[str, str]:
    manager = _manager(request)
    return await asyncio.to_thread(manager.stop_training)


@router.get("/model/status")
async def model_status(request: Request) -> dict[str, Any]:
    manager = _manager(request)
    return await asyncio.to_thread(manager.training_status)


@router.get("/model/metrics")
async def model_metrics(request: Request) -> dict[str, Any]:
    manager = _manager(request)
    return await asyncio.to_thread(manager.model_metrics)


@router.post("/model/reload")
async def model_reload(request: Request) -> dict[str, Any]:
    service = request.app.state.prediction_service
    await asyncio.to_thread(service.reload)
    return {"available": service.available, "error": service.error}
