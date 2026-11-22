"""Label ordering and category helpers shared by training and inference."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePath
from typing import Sequence


_ALPHABET_LABELS = {chr(code) for code in range(ord("A"), ord("Z") + 1)} | {"E1", "E2"}
_NUMERAL_LABELS = {str(digit) for digit in range(10)} | {"9A", "9B"}
_HINDI_VOWELS = {"अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ"}


def canonicalize_label(label: str) -> str:
    """Validate and preserve the original dataset label exactly."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Class labels cannot be empty")
    return label


def category_for_label(label: str) -> str:
    """Return a conservative category without inventing label semantics."""
    value = canonicalize_label(label).strip()
    upper = value.upper()
    if upper in _ALPHABET_LABELS:
        return "alphabet"
    if upper in _NUMERAL_LABELS:
        return "numeral"
    if value in _HINDI_VOWELS:
        return "hindi_vowel"
    return "other"


def is_known_mendeley_label(value: str) -> bool:
    """Check the documented Mendeley ISL label families (A-Z, E1/E2, 0-9, 9A/9B, vowels)."""
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    if candidate in _HINDI_VOWELS:
        return True
    upper = candidate.upper()
    return upper in _ALPHABET_LABELS or upper in _NUMERAL_LABELS


def find_label_component(parts: tuple[str, ...] | list[str]) -> str | None:
    """Return the nearest non-empty path component; dataset discovery is structural."""
    for part in reversed(parts):
        if part.strip():
            return canonicalize_label(part)
    return None


def natural_class_names(labels: Sequence[str]) -> list[str]:
    """Return deterministic, case-sensitive unique labels in natural order."""
    canonical = [canonicalize_label(label) for label in labels]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Class labels must be unique after normalization")
    if any("\x00" in label or "/" in label or "\\" in label for label in canonical):
        raise ValueError("Class labels cannot contain path separators or NUL bytes")

    def key(label: str) -> tuple[object, ...]:
        return tuple(
            (1, int(piece)) if piece.isdigit() else (0, piece.casefold())
            for piece in re.split(r"(\d+)", label)
        ) + ((2, label),)

    return sorted(canonical, key=key)


def validate_class_names(class_names: Sequence[str]) -> list[str]:
    names = list(class_names)
    if len(names) < 2:
        raise ValueError("At least two classes are required")
    ordered = natural_class_names(names)
    if set(ordered) != set(names) or len(ordered) != len(names):
        raise ValueError("Class names must be unique, non-empty labels")
    return names


def label_mapping(class_names: Sequence[str]) -> dict[str, int]:
    names = validate_class_names(class_names)
    return {label: index for index, label in enumerate(names)}


def write_labels(path: Path, class_names: Sequence[str]) -> None:
    names = validate_class_names(class_names)
    payload = {
        "class_names": names,
        "label_mapping": label_mapping(names),
        "categories": {name: category_for_label(name) for name in names},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_labels(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read label metadata: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("class_names"), list):
        raise ValueError(f"Label metadata must contain class_names: {path}")
    names = payload["class_names"]
    if not all(isinstance(name, str) for name in names):
        raise ValueError(f"Label metadata contains a non-string class: {path}")
    names = validate_class_names(names)
    if payload.get("label_mapping") != label_mapping(names):
        raise ValueError(f"Label mapping does not match class ordering: {path}")
    return names


def display_path(path: PurePath, root: PurePath | None = None) -> str:
    if root is not None:
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return path.as_posix()
