from __future__ import annotations

import json

import pytest

from ml.evaluation import classification_metrics, save_evaluation


def test_classification_metrics_reports_accuracy_and_macro_scores():
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], ["A", "B"])
    assert metrics["total"] == 4
    assert metrics["correct"] == 3
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision_macro"] == pytest.approx(0.833333, abs=1e-5)
    assert metrics["recall_macro"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"]["labels"] == ["A", "B"]
    assert metrics["confusion_matrix"]["values"] == [[2, 0], [1, 1]]
    assert len(metrics["classification_report"]) == 2


def test_classification_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0], ["A", "B"])


def test_save_evaluation_writes_json_and_csv(tmp_path):
    metrics = classification_metrics([0, 1], [0, 1], ["A", "B"])
    paths = save_evaluation(tmp_path, metrics, ["A", "B"], prefix="static")
    assert json.loads((tmp_path / "static_metrics.json").read_text(encoding="utf-8"))["accuracy"] == 1.0
    assert (tmp_path / "static_report.csv").exists()
    assert (tmp_path / "static_confusion.csv").exists()
    assert paths["json"].endswith("static_metrics.json")
