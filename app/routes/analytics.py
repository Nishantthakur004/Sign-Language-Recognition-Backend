from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_db, get_optional_user
from app.schemas.recognition import AnalyticsOverview, PopularSignsResponse, TrendsResponse

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverview)
async def overview(user: dict[str, Any] = Depends(get_optional_user), db: Any = Depends(get_db)) -> dict[str, Any]:
    filters = {"user_id": user["_id"]}
    total = await db.recognition_history.count_documents(filters)
    aggregate = await db.recognition_history.aggregate([
        {"$match": filters},
        {"$group": {"_id": None, "average": {"$avg": "$confidence"}, "duration": {"$sum": "$duration_ms"}}},
    ]).to_list(length=1)
    sessions = await db.recognition_history.distinct("session_id", filters)
    duration = int(aggregate[0].get("duration") or 0) if aggregate else 0
    return {
        "total_recognitions": total,
        "average_confidence": round(float(aggregate[0].get("average") or 0), 6) if aggregate else 0.0,
        "total_sessions": len({session for session in sessions if session}),
        "total_duration_ms": duration,
        "total_duration_seconds": round(duration / 1000, 3),
    }


@router.get("/trends", response_model=TrendsResponse)
async def trends(
    period: Literal["daily", "weekly", "monthly"] = Query(default="daily"),
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, Any]:
    date_format = {"daily": "%Y-%m-%d", "weekly": "%G-W%V", "monthly": "%Y-%m"}[period]
    rows = await db.recognition_history.aggregate([
        {"$match": {"user_id": user["_id"]}},
        {"$group": {
            "_id": {"$dateToString": {"format": date_format, "date": "$created_at"}},
            "count": {"$sum": 1},
            "average_confidence": {"$avg": "$confidence"},
        }},
        {"$sort": {"_id": 1}},
    ]).to_list(length=None)
    return {
        "period": period,
        "items": [
            {
                "period": row["_id"],
                "count": row["count"],
                "average_confidence": round(float(row.get("average_confidence") or 0), 6),
            }
            for row in rows
        ],
    }


@router.get("/popular-signs", response_model=PopularSignsResponse)
async def popular_signs(
    limit: int = Query(default=10, ge=1, le=100),
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, Any]:
    rows = await db.recognition_history.aggregate([
        {"$match": {"user_id": user["_id"]}},
        {"$group": {"_id": "$sign", "count": {"$sum": 1}, "average_confidence": {"$avg": "$confidence"}}},
        {"$sort": {"count": -1, "_id": 1}},
        {"$limit": limit},
    ]).to_list(length=limit)
    return {
        "items": [
            {
                "sign": row["_id"],
                "count": row["count"],
                "average_confidence": round(float(row.get("average_confidence") or 0), 6),
            }
            for row in rows
        ]
    }
