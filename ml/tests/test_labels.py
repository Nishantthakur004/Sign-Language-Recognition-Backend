from __future__ import annotations

import pytest

from ml.labels import (
    canonicalize_label,
    category_for_label,
    is_known_mendeley_label,
    label_mapping,
    natural_class_names,
    read_labels,
    write_labels,
)


def test_canonicalize_preserves_original_label():
    assert canonicalize_label("9a") == "9a"
    assert canonicalize_label("E1") == "E1"
    with pytest.raises(ValueError):
        canonicalize_label("   ")


def test_categories_cover_dataset_families():
    assert category_for_label("A") == "alphabet"
    assert category_for_label("E1") == "alphabet"
    assert category_for_label("7") == "numeral"
    assert category_for_label("9b") == "numeral"
    assert category_for_label("अ") == "hindi_vowel"
    assert category_for_label("Hello") == "other"


def test_known_label_validation_rejects_unknown_values():
    assert is_known_mendeley_label("A")
    assert is_known_mendeley_label("e2")
    assert is_known_mendeley_label("9a")
    assert is_known_mendeley_label("अ")
    assert not is_known_mendeley_label("Hello")
    assert not is_known_mendeley_label("")
    assert not is_known_mendeley_label(None)


def test_natural_class_names_orders_numerals_numerically():
    assert natural_class_names(["10", "2", "A", "1"]) == ["1", "2", "10", "A"]


def test_label_mapping_and_round_trip(tmp_path):
    names = ["A", "B", "1"]
    assert label_mapping(names) == {"A": 0, "B": 1, "1": 2}
    path = tmp_path / "labels.json"
    write_labels(path, names)
    assert read_labels(path) == names
