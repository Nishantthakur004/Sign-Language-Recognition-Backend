"""Evaluation metrics and persisted reports for static and dynamic models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for evaluation. Install backend/requirements.txt first."
        ) from exc


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute accuracy, macro precision/recall/F1, per-class stats, and a confusion matrix."""
    _require_sklearn()
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    truth = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if truth.size == 0:
        raise ValueError("Cannot compute metrics from an empty label set")

    labels = list(class_names)
    present = {int(value) for value in truth} | {int(value) for value in predicted}
    metric_labels = sorted(set(range(len(labels))) | present)

    per_class: list[dict[str, Any]] = []
    for label in metric_labels:
        mask = truth == label
        support = int(mask.sum())
        tp = int(((truth == label) & (predicted == label)).sum())
        fp = int(((truth != label) & (predicted == label)).sum())
        fn = int(((truth == label) & (predicted != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        name = labels[label] if 0 <= label < len(labels) else str(label)
        per_class.append(
            {
                "class": name,
                "index": label,
                "support": support,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )

    cm = confusion_matrix(truth, predicted, labels=metric_labels)
    report_rows = _classification_report_rows(truth, predicted, metric_labels, labels)

    return {
        "total": int(truth.size),
        "correct": int((truth == predicted).sum()),
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "precision_macro": round(float(precision_score(truth, predicted, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(truth, predicted, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [labels[label] if 0 <= label < len(labels) else str(label) for label in metric_labels],
            "values": cm.tolist(),
        },
        "classification_report": report_rows,
    }


def _classification_report_rows(
    truth: np.ndarray,
    predicted: np.ndarray,
    metric_labels: Sequence[int],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in metric_labels:
        mask = truth == label
        support = int(mask.sum())
        tp = int(((truth == label) & (predicted == label)).sum())
        fp = int(((truth != label) & (predicted == label)).sum())
        fn = int(((truth == label) & (predicted != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "class": labels[label] if 0 <= label < len(labels) else str(label),
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1": f"{f1:.4f}",
                "support": support,
            }
        )
    return rows


def save_evaluation(
    directory: str | Path,
    metrics: dict[str, Any],
    class_names: Sequence[str],
    *,
    prefix: str = "model",
) -> dict[str, str]:
    """Persist evaluation outputs: JSON summary, classification report CSV, confusion matrix CSV/PNG."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}_metrics.json"
    json_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report_path = output_dir / f"{prefix}_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for row in metrics.get("classification_report", []):
            writer.writerow([row["class"], row["precision"], row["recall"], row["f1"], row["support"]])

    cm = metrics.get("confusion_matrix")
    cm_csv_path = output_dir / f"{prefix}_confusion.csv"
    png_path: str | None = None
    if cm:
        with cm_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true"] + list(cm.get("labels", [])))
            for index, row in enumerate(cm.get("values", [])):
                writer.writerow([list(cm.get("labels", []))[index]] + row)
        png_path = _render_confusion_png(cm, output_dir / f"{prefix}_confusion.png")

    return {
        "json": str(json_path),
        "report": str(report_path),
        "confusion_csv": str(cm_csv_path) if cm else "",
        "confusion_png": png_path or "",
    }


def _render_confusion_png(cm: dict[str, Any], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = cm.get("labels", [])
    values = np.asarray(cm.get("values", []))
    figure, axis = plt.subplots(figsize=(max(4, len(labels) * 0.4), max(4, len(labels) * 0.4)))
    axis.imshow(values, interpolation="nearest", cmap=matplotlib.colormaps.get_cmap("Blues"))
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90)
    axis.set_yticklabels(labels)
    threshold = values.max() / 2 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > threshold else "black"
            axis.text(j, i, int(values[i, j]), ha="center", va="center", color=color)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)
