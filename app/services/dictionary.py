from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.config import Settings

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SignEntry:
    id: str
    label: str
    category: str
    image_path: Path | None

    def public(self, favourite: bool = False) -> dict[str, Any]:
        has_image = self.image_path is not None and self.image_path.is_file()
        return {
            "id": self.id,
            "name": self.label,
            "meaning": self.label,
            "category": self.category,
            "language": "dataset",
            "description": f"Sign dictionary entry for the dataset label '{self.label}'.",
            "video_url": None,
            "related_signs": [],
            "image_available": has_image,
            "image_url": f"/api/dictionary/{quote(self.id, safe='')}/image" if has_image else None,
            "demo_data": False,
            "is_favourite": favourite,
        }


HINDI_VOWELS = {"अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ"}


def category_for_label(label: str, fallback: str = "sign") -> str:
    value = label.strip()
    if re.fullmatch(r"(?:[A-Z]|E[12])", value, re.IGNORECASE):
        return "alphabet"
    if re.fullmatch(r"(?:[0-9]|9[AB])", value, re.IGNORECASE):
        return "numeral"
    if value in HINDI_VOWELS or value.casefold().startswith("hindi_vowel"):
        return "hindi_vowel"
    return fallback.strip() or "sign"


class DictionaryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache_signature: tuple[Any, ...] | None = None
        self._cached_entries: dict[str, SignEntry] = {}

    def _candidate_files(self) -> list[Path]:
        candidates = [
            self.settings.dataset_report,
            self.settings.models_dir / "dataset_report.json",
            self.settings.dataset_root / "dataset_report.json",
            self.settings.dataset_root / "manifest.json",
            self.settings.dataset_root / "dataset_manifest.json",
            self.settings.dataset_root / "processed_manifest.json",
        ]
        return list(dict.fromkeys(path.resolve() for path in candidates))

    def _metadata_signature(self) -> tuple[Any, ...]:
        signature = []
        for path in self._candidate_files():
            try:
                stat = path.stat()
            except OSError:
                signature.append((str(path), None))
            else:
                signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def _safe_image(self, value: Any) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.settings.dataset_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.settings.dataset_root.resolve())
        except (OSError, RuntimeError, ValueError):
            return None
        if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_SUFFIXES:
            return None
        return resolved

    def _read_entries(self) -> dict[str, SignEntry]:
        entries: dict[str, SignEntry] = {}

        def add(label: Any, record: Any = None) -> None:
            if not isinstance(label, str) or not label.strip():
                return
            label = label.strip()
            image = None
            if isinstance(record, dict):
                for key in ("sample_image", "image_path", "file_path", "path", "sample"):
                    image = self._safe_image(record.get(key))
                    if image is not None:
                        break
            entry_id = hashlib.sha256(label.encode("utf-8")).hexdigest()[:24]
            existing = entries.get(entry_id)
            record_category = record.get("category") if isinstance(record, dict) else None
            fallback = record_category.strip() if isinstance(record_category, str) else "sign"
            entries[entry_id] = SignEntry(
                id=entry_id,
                label=label,
                category=category_for_label(label, fallback),
                image_path=existing.image_path if existing and existing.image_path else image,
            )

        def visit(value: Any, key: str | None = None) -> None:
            if isinstance(value, list):
                if key in {"labels", "classes", "class_names"} and all(isinstance(item, str) for item in value):
                    for label in value:
                        add(label)
                    return
                if key in {"samples", "items", "images", "files", "records", "manifest"} or key is None:
                    for item in value:
                        if isinstance(item, dict):
                            add(item.get("label", item.get("class_name", item.get("sign"))), item)
                return
            if not isinstance(value, dict):
                return
            for label_key in ("label", "class_name", "sign"):
                if label_key in value:
                    add(value[label_key], value)
                    break
            for child_key, child in value.items():
                if child_key in {"labels", "class_names"}:
                    visit(child, child_key)
                elif child_key == "classes":
                    if isinstance(child, dict):
                        for label, details in child.items():
                            add(label, details)
                    else:
                        visit(child, child_key)
                elif child_key in {"class_distribution", "label_counts", "counts"} and isinstance(child, dict):
                    for label, details in child.items():
                        add(label, details)
                elif child_key in {"samples", "items", "images", "files", "records", "manifest", "train", "validation", "val", "test", "splits"}:
                    visit(child, child_key)

        for path in self._candidate_files():
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    visit(json.load(handle))
            except (OSError, UnicodeError, json.JSONDecodeError):
                logger.warning("Ignoring unreadable dataset metadata: %s", path)
        return entries

    def _entries(self) -> dict[str, SignEntry]:
        signature = self._metadata_signature()
        if signature != self._cache_signature:
            self._cached_entries = self._read_entries()
            self._cache_signature = signature
        return self._cached_entries

    def entries(self) -> list[SignEntry]:
        return sorted(self._entries().values(), key=lambda entry: (entry.label.casefold(), entry.label))

    def get(self, sign_id: str) -> SignEntry | None:
        return self._entries().get(sign_id)

    def dataset_info(self) -> dict[str, Any]:
        entries = self.entries()
        category_counts: dict[str, int] = {}
        for entry in entries:
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
        return {
            "available": bool(entries),
            "total_labels": len(entries),
            "categories": sorted(category_counts),
            "category_counts": category_counts,
            "metadata_files": [path.name for path in self._candidate_files() if path.is_file()],
        }
