from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_SEQUENCE_FRAMES = 240
MAX_LANDMARK_VALUES = 50_000


def normalize_landmark_data(value: Any, *, sequence: bool = False) -> Any:
    """Validate bounded, finite numeric landmark data without fixing its frame dimensions."""
    count = 0

    def normalize(item: Any, depth: int) -> Any:
        nonlocal count
        if depth > 5:
            raise ValueError("Landmark data is nested too deeply")
        if isinstance(item, bool) or item is None:
            raise ValueError("Landmark values must be finite numbers")
        if isinstance(item, (int, float)):
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("Landmark values must be finite numbers")
            count += 1
            if count > MAX_LANDMARK_VALUES:
                raise ValueError("Landmark data contains too many values")
            return number
        if isinstance(item, list):
            if len(item) > MAX_LANDMARK_VALUES:
                raise ValueError("Landmark data contains too many values")
            return [normalize(child, depth + 1) for child in item]
        if isinstance(item, dict):
            if len(item) > 100:
                raise ValueError("Landmark objects contain too many fields")
            return {str(key): normalize(child, depth + 1) for key, child in item.items()}
        raise ValueError("Landmarks must contain only numbers, arrays, and objects")

    if not isinstance(value, (list, dict)) or not value:
        raise ValueError("Landmark data must be a non-empty array or object")
    if sequence and (not isinstance(value, list) or len(value) > MAX_SEQUENCE_FRAMES):
        raise ValueError(f"Sequences must contain 1 to {MAX_SEQUENCE_FRAMES} frames")
    return normalize(value, 0)


def parse_landmark_json(value: str | None, *, sequence: bool = False) -> Any:
    if value is None:
        return None
    if len(value) > 2_000_000:
        raise ValueError("Landmark JSON is too large")

    def reject_constant(_: str) -> None:
        raise ValueError("Landmark values must be finite numbers")

    try:
        decoded = json.loads(value, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Landmark data must be valid JSON") from exc
    return normalize_landmark_data(decoded, sequence=sequence)


class PredictionResult(BaseModel):
    prediction_id: str | None = None
    prediction_token: str | None = None
    sign: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)
    category: str | None = Field(default=None, max_length=100)
    model_version: str | None = Field(default=None, max_length=100)
    recognition_type: str | None = Field(default=None, max_length=100)
    top_k: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    hand_detected: bool | None = None
    landmarks: Any = None


class HistorySaveRequest(BaseModel):
    prediction_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=100)
    thumbnail: str | None = Field(default=None, max_length=20_000_000)
    duration_ms: int = Field(default=0, ge=0, le=86_400_000)

    @field_validator("prediction_id")
    @classmethod
    def clean_prediction_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prediction ID cannot be blank")
        return value

    @field_validator("session_id")
    @classmethod
    def clean_session(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class HistoryResponse(BaseModel):
    id: str
    prediction_id: str | None = None
    sign: str
    confidence: float
    category: str | None = None
    model_version: str | None = None
    recognition_type: str | None = None
    session_id: str | None = None
    duration_ms: int = 0
    thumbnail: str | None = None
    created_at: datetime | None = None


class HistoryListResponse(BaseModel):
    """History list response: {items, total, page, page_size}."""

    items: list[HistoryResponse]
    total: int
    page: int
    page_size: int


class AnalyticsOverview(BaseModel):
    total_recognitions: int
    average_confidence: float
    total_sessions: int
    total_duration_ms: int
    total_duration_seconds: float


class TrendPoint(BaseModel):
    period: str
    count: int
    average_confidence: float


class TrendsResponse(BaseModel):
    period: Literal["daily", "weekly", "monthly"]
    items: list[TrendPoint]


class PopularSignsResponse(BaseModel):
    items: list[dict[str, Any]]
