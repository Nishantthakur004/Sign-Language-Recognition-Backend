from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.dependencies import get_db, get_optional_user
from app.schemas.dictionary import CategoriesResponse, FavouriteRequest, SignListResponse, SignResponse

router = APIRouter(prefix="/api/dictionary", tags=["dictionary"])


def filtered_entries(request: Request, search: str | None, category: str | None, letter: str | None) -> list[Any]:
    entries = request.app.state.dictionary_service.entries()
    if search and search.strip():
        term = search.strip().casefold()
        entries = [entry for entry in entries if term in entry.label.casefold() or term in entry.category.casefold()]
    if category and category.strip():
        wanted = category.strip().casefold()
        entries = [entry for entry in entries if entry.category.casefold() == wanted]
    if letter and letter.strip():
        initial = letter.strip().casefold()
        entries = [entry for entry in entries if entry.label.casefold().startswith(initial)]
    return entries


def page_entries(entries: list[Any], page: int, page_size: int, favourites: set[str] | None = None) -> dict[str, Any]:
    favourites = favourites or set()
    start = (page - 1) * page_size
    return {
        "items": [entry.public(entry.id in favourites) for entry in entries[start:start + page_size]],
        "total": len(entries),
        "page": page,
        "page_size": page_size,
    }


@router.get("", response_model=SignListResponse)
async def list_dictionary(
    request: Request,
    search: str | None = Query(default=None, max_length=100),
    category: str | None = Query(default=None, max_length=100),
    letter: str | None = Query(default=None, max_length=20),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    return page_entries(filtered_entries(request, search, category, letter), page, page_size)


@router.get("/search", response_model=SignListResponse)
async def search_dictionary(
    request: Request,
    q: str | None = Query(default=None, max_length=100),
    category: str | None = Query(default=None, max_length=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    return page_entries(filtered_entries(request, q, category, None), page, page_size)


@router.get("/categories", response_model=CategoriesResponse)
async def categories(request: Request) -> dict[str, Any]:
    return {"categories": sorted({entry.category for entry in request.app.state.dictionary_service.entries()})}


@router.get("/favourites", response_model=SignListResponse)
async def list_favourites(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, Any]:
    favourites = await db.favourites.find({"user_id": user["_id"]}).to_list(length=None)
    favourite_ids = {item["sign_id"] for item in favourites}
    entries = [entry for entry in request.app.state.dictionary_service.entries() if entry.id in favourite_ids]
    return page_entries(entries, page, page_size, favourite_ids)


@router.post("/favourites", response_model=SignResponse, status_code=201)
async def add_favourite(
    payload: FavouriteRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, Any]:
    entry = request.app.state.dictionary_service.get(payload.sign_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dictionary sign not found")
    await db.favourites.update_one(
        {"user_id": user["_id"], "sign_id": entry.id},
        {"$setOnInsert": {"user_id": user["_id"], "sign_id": entry.id}},
        upsert=True,
    )
    return entry.public(True)


@router.delete("/favourites/{sign_id}")
async def remove_favourite(
    sign_id: str,
    user: dict[str, Any] = Depends(get_optional_user),
    db: Any = Depends(get_db),
) -> dict[str, str]:
    result = await db.favourites.delete_one({"user_id": user["_id"], "sign_id": sign_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Favourite not found")
    return {"message": "Favourite removed"}


@router.get("/{sign_id}/image")
async def dictionary_image(sign_id: str, request: Request) -> FileResponse:
    entry = request.app.state.dictionary_service.get(sign_id)
    if entry is None or entry.image_path is None or not entry.image_path.is_file():
        raise HTTPException(status_code=404, detail="No sample image is available for this sign")
    media_type = "image/png" if entry.image_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(entry.image_path, media_type=media_type, filename=f"sample{entry.image_path.suffix.lower()}")


@router.get("/{sign_id}", response_model=SignResponse)
async def dictionary_detail(sign_id: str, request: Request) -> dict[str, Any]:
    entry = request.app.state.dictionary_service.get(sign_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dictionary sign not found")
    return entry.public()
