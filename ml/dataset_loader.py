"""Inspect the original ISL dataset, remove bad/duplicate images, and build leakage-resistant splits.

The Mendeley ISL dataset is organised as::

    <root>/<age group>/<sleeve session>/<label>/<image>.jpeg

Labels are preserved exactly as they appear in the dataset (including ``E1``/``E2``
and ``9a``/``9b``).  The age group and sleeve session are treated as recording
groups so that images from the same session never appear in more than one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STRUCTURAL_DIRECTORIES = {
    "kids",
    "teenagers",
    "adults",
    "full sleeves",
    "half sleeves",
    "full_sleeves",
    "half_sleeves",
    "train",
    "test",
    "validation",
    "val",
}
DEFAULT_SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be inspected or split safely."""


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str
    group: str
    split: str
    sha256: str | None = None
    width: int | None = None
    height: int | None = None


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _label_for(path: Path, root: Path) -> tuple[str, str]:
    """Return (label, group) from a dataset path without renaming the label."""
    relative = path.relative_to(root)
    parts = list(relative.parts[:-1])
    if not parts:
        raise DatasetError(f"Image is not inside a label directory: {path}")
    label = parts[-1].strip()
    if not label:
        raise DatasetError(f"Empty label directory for image: {path}")
    group_parts = [part.strip() for part in parts[:-1] if part.strip()]
    group = "/".join(group_parts) if group_parts else "default"
    return label, group


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise DatasetError(f"Corrupt or unsupported image: {path}") from exc
    if width < 1 or height < 1:
        raise DatasetError(f"Image has invalid dimensions: {path}")
    return width, height


def inspect_dataset(
    root: str | Path,
    *,
    validate: bool = True,
    deduplicate: bool = True,
) -> tuple[list[ImageRecord], dict[str, Any]]:
    """Walk the dataset, validate images, and remove exact duplicates.

    Returns records without a split plus a report containing every rejected file.
    """
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {dataset_root}")

    candidates = sorted(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda item: item.as_posix().casefold(),
    )
    if not candidates:
        raise DatasetError(f"No supported images found under {dataset_root}")

    corrupt: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    records: list[ImageRecord] = []
    sizes: dict[int, list[Path]] = defaultdict(list)
    for path in candidates:
        try:
            sizes[path.stat().st_size].append(path)
        except OSError as exc:
            corrupt.append({"path": _relative(path, dataset_root), "reason": str(exc)})

    hashes: dict[str, str] = {}
    if deduplicate:
        for same_size in sizes.values():
            if len(same_size) < 2:
                continue
            for path in same_size:
                try:
                    hashes[_relative(path, dataset_root)] = _hash_file(path)
                except OSError as exc:
                    corrupt.append({"path": _relative(path, dataset_root), "reason": str(exc)})

    seen_hashes: dict[str, str] = {}
    for path in candidates:
        relative = _relative(path, dataset_root)
        if any(item["path"] == relative for item in corrupt):
            continue
        try:
            label, group = _label_for(path, dataset_root)
        except DatasetError as exc:
            corrupt.append({"path": relative, "reason": str(exc)})
            continue
        digest = hashes.get(relative)
        if digest is not None:
            if digest in seen_hashes:
                duplicates.append(
                    {"path": relative, "duplicate_of": seen_hashes[digest], "sha256": digest}
                )
                continue
            seen_hashes[digest] = relative
        width = height = None
        if validate:
            try:
                width, height = _validate_image(path)
            except DatasetError as exc:
                corrupt.append({"path": relative, "reason": str(exc)})
                continue
        records.append(
            ImageRecord(
                path=relative,
                label=label,
                group=group,
                split="",
                sha256=digest,
                width=width,
                height=height,
            )
        )

    if not records:
        raise DatasetError("Every image was rejected; check the dataset directory and formats")

    distribution = Counter(record.label for record in records)
    groups = sorted({record.group for record in records})
    report = {
        "dataset": "ISL Static Gesture Dataset (Mendeley 7tsw22y96w)",
        "root": str(dataset_root),
        "total_files": len(candidates),
        "valid_images": len(records),
        "classes": sorted(distribution),
        "class_distribution": dict(sorted(distribution.items())),
        "groups": groups,
        "group_count": len(groups),
        "corrupt_images": corrupt,
        "duplicate_images": duplicates,
        "label_mapping": {label: label for label in sorted(distribution)},
        "validated": validate,
        "deduplicated": deduplicate,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return records, report


def _split_groups(
    records: Sequence[ImageRecord],
    splits: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits, keeping every class in the training split."""
    groups = sorted({record.group for record in records})
    if len(groups) < 2:
        raise DatasetError(
            "Only one recording group was found. Group-aware splitting is impossible; "
            "provide subject/session folders or use --allow-ungrouped-split explicitly."
        )
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_count = max(1, round(total * splits["test"]))
    validation_count = max(1, round(total * splits["validation"]))
    if test_count + validation_count >= total:
        test_count = 1
        validation_count = 1
    assignments: dict[str, str] = {}
    for index, group in enumerate(shuffled):
        if index < test_count:
            assignments[group] = "test"
        elif index < test_count + validation_count:
            assignments[group] = "validation"
        else:
            assignments[group] = "train"

    train_labels = {record.label for record in records if assignments[record.group] == "train"}
    all_labels = {record.label for record in records}
    missing = sorted(all_labels - train_labels)
    if missing:
        raise DatasetError(
            "Group-aware split left classes without training examples: " + ", ".join(missing)
        )
    return assignments


def _split_ungrouped(
    records: Sequence[ImageRecord],
    splits: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Fallback stratified split when the dataset has no session metadata."""
    rng = random.Random(seed)
    by_label: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    assignments: dict[str, str] = {}
    for label, items in sorted(by_label.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        count = len(shuffled)
        test_count = max(1, round(count * splits["test"])) if count >= 3 else 0
        validation_count = max(1, round(count * splits["validation"])) if count >= 3 else 0
        if test_count + validation_count >= count:
            test_count = 1
            validation_count = 0
        for index, record in enumerate(shuffled):
            if index < test_count:
                assignments[record.path] = "test"
            elif index < test_count + validation_count:
                assignments[record.path] = "validation"
            else:
                assignments[record.path] = "train"
    return assignments


def split_records(
    records: Sequence[ImageRecord],
    *,
    splits: dict[str, float] | None = None,
    seed: int = 42,
    allow_ungrouped: bool = False,
) -> tuple[list[ImageRecord], dict[str, Any]]:
    """Return records with split labels and a split summary."""
    splits = splits or DEFAULT_SPLITS
    if abs(sum(splits.values()) - 1.0) > 1e-6:
        raise DatasetError("Split fractions must sum to 1.0")
    groups = {record.group for record in records}
    group_aware = len(groups) >= 2
    if group_aware:
        assignments = _split_groups(records, splits, seed)
        split_for = lambda record: assignments[record.group]
    else:
        if not allow_ungrouped:
            raise DatasetError(
                "The dataset has no recording groups. Re-run with --allow-ungrouped-split "
                "to use a stratified image-level split (weaker leakage protection)."
            )
        assignments = _split_ungrouped(records, splits, seed)
        split_for = lambda record: assignments[record.path]

    result = [
        ImageRecord(
            path=record.path,
            label=record.label,
            group=record.group,
            split=split_for(record),
            sha256=record.sha256,
            width=record.width,
            height=record.height,
        )
        for record in records
    ]
    summary = {
        "group_aware": group_aware,
        "seed": seed,
        "splits": {
            split: {
                "images": sum(1 for record in result if record.split == split),
                "classes": len({record.label for record in result if record.split == split}),
                "groups": sorted({record.group for record in result if record.split == split}),
            }
            for split in ("train", "validation", "test")
        },
    }
    return result, summary


def build_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    report_path: str | Path | None = None,
    seed: int = 42,
    validate: bool = True,
    allow_ungrouped: bool = False,
) -> dict[str, Any]:
    """Inspect, split, and write ``manifest.json`` plus ``dataset_report.json``."""
    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records, report = inspect_dataset(root, validate=validate)
    records, split_summary = split_records(
        records,
        seed=seed,
        allow_ungrouped=allow_ungrouped,
    )
    report.update(split_summary)
    samples: dict[str, str] = {}
    for record in records:
        samples.setdefault(record.label, record.path)
    report["samples"] = dict(sorted(samples.items()))
    report["sample_images"] = [
        {"label": label, "sample_image": path} for label, path in sorted(samples.items())
    ]
    report["manifest"] = str(destination / "manifest.json")
    report["total_images"] = len(records)

    manifest = {
        "dataset": report["dataset"],
        "root": str(root),
        "created_at": report["created_at"],
        "classes": report["classes"],
        "records": [asdict(record) for record in records],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_file = Path(report_path).expanduser().resolve() if report_path else destination / "dataset_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read dataset manifest: {manifest_path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise DatasetError("Manifest must contain a records array")
    return payload


def records_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    records = [record for record in manifest["records"] if record.get("split") == split]
    if not records:
        raise DatasetError(f"Manifest has no records for the {split} split")
    return records


def resolve_record_path(manifest: dict[str, Any], record: dict[str, Any]) -> Path:
    return Path(manifest["root"]) / record["path"]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the ISL dataset and build a split manifest")
    parser.add_argument("--dataset-root", required=True, help="Path to the extracted Mendeley dataset")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "datasets" / "processed"), help="Directory for manifest.json")
    parser.add_argument("--report", default=str(Path(__file__).resolve().parent / "models" / "dataset_report.json"), help="Dataset report path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true", help="Skip full image decoding")
    parser.add_argument("--allow-ungrouped-split", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = build_manifest(
        args.dataset_root,
        args.output_dir,
        report_path=args.report,
        seed=args.seed,
        validate=not args.skip_validation,
        allow_ungrouped=args.allow_ungrouped_split,
    )
    logger.info(
        "Prepared %s images across %s classes; corrupt=%s duplicates=%s",
        report["total_images"],
        len(report["classes"]),
        len(report["corrupt_images"]),
        len(report["duplicate_images"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
