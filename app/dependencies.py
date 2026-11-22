from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request


def get_db(request: Request) -> Any:
    db = getattr(request.app.state, "db", None)
    if (
        db is None
        or not getattr(request.app.state, "db_available", False)
        or not getattr(request.app.state, "db_indexes_ready", False)
    ):
        raise HTTPException(status_code=503, detail="Database is unavailable")
    return db


GUEST_USER: dict[str, Any] = {
    "_id": "guest",
    "name": "Guest",
    "email": None,
    "role": "guest",
    "is_active": True,
}


async def get_optional_user() -> dict[str, Any]:
    """The app is guest-only; every request is attributed to the guest profile."""
    return GUEST_USER
