from __future__ import annotations

from pydantic import BaseModel, Field


class FavouriteRequest(BaseModel):
    sign_id: str = Field(min_length=1, max_length=200)


class SignResponse(BaseModel):
    id: str
    name: str
    meaning: str
    category: str
    language: str
    description: str
    video_url: str | None = None
    related_signs: list[str] = Field(default_factory=list)
    image_available: bool
    image_url: str | None = None
    demo_data: bool = False
    is_favourite: bool = False


class SignListResponse(BaseModel):
    """Dictionary list response: {items, total, page, page_size}."""

    items: list[SignResponse]
    total: int
    page: int
    page_size: int


class CategoriesResponse(BaseModel):
    categories: list[str]
