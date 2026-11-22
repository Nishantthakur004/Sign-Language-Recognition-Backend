from __future__ import annotations

import json

import pytest
from PIL import Image

from ml.dataset_loader import (
    DatasetError,
    ImageRecord,
    build_manifest,
    inspect_dataset,
    load_manifest,
    records_for_split,
    split_records,
)


def _write_image(path, color=(120, 80, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="JPEG")


@pytest.fixture()
def dataset_root(tmp_path):
    root = tmp_path / "dataset"
    _write_image(root / "Kids" / "Full Sleeves" / "A" / "a1.jpg")
    _write_image(root / "Kids" / "Full Sleeves" / "A" / "a2.jpg", (10, 20, 30))
    _write_image(root / "Kids" / "Full Sleeves" / "B" / "b1.jpg", (40, 50, 60))
    _write_image(root / "Teenagers" / "Half Sleeves" / "A" / "a3.jpg", (70, 80, 90))
    _write_image(root / "Teenagers" / "Half Sleeves" / "B" / "b2.jpg", (100, 110, 120))
    _write_image(root / "Adults" / "Full Sleeves" / "A" / "a4.jpg", (130, 140, 150))
    _write_image(root / "Adults" / "Full Sleeves" / "B" / "b3.jpg", (160, 170, 180))
    # Exact duplicate of a1.jpg
    (root / "Kids" / "Full Sleeves" / "A" / "a1_copy.jpg").write_bytes(
        (root / "Kids" / "Full Sleeves" / "A" / "a1.jpg").read_bytes()
    )
    # Corrupt image
    (root / "Kids" / "Full Sleeves" / "A" / "broken.jpg").write_bytes(b"not an image")
    return root


def test_inspect_dataset_removes_duplicates_and_corrupt_files(dataset_root):
    records, report = inspect_dataset(dataset_root)
    assert report["total_files"] == 9
    assert report["valid_images"] == 7
    assert len(report["duplicate_images"]) == 1
    assert len(report["corrupt_images"]) == 1
    assert report["classes"] == ["A", "B"]
    assert report["group_count"] == 3
    assert all(record.sha256 for record in records)


def test_build_manifest_writes_group_aware_splits(dataset_root, tmp_path):
    output = tmp_path / "processed"
    report = build_manifest(dataset_root, output, report_path=tmp_path / "report.json")
    assert report["group_aware"] is True
    assert report["total_images"] == 7
    assert sum(report["splits"][split]["images"] for split in ("train", "validation", "test")) == 7
    manifest = load_manifest(output / "manifest.json")
    assert manifest["classes"] == ["A", "B"]
    assert len(manifest["records"]) == 7
    assert records_for_split(manifest, "train")
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["total_images"] == 7


def test_single_group_requires_explicit_opt_in():
    records = [
        ImageRecord(path="a.jpg", label="A", group="default", split=""),
        ImageRecord(path="b.jpg", label="B", group="default", split=""),
    ]
    with pytest.raises(DatasetError):
        split_records(records)
    split, summary = split_records(records, allow_ungrouped=True)
    assert summary["group_aware"] is False
    assert {record.split for record in split} <= {"train", "validation", "test"}
