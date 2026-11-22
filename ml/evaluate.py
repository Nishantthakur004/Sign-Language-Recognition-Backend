"""Evaluate saved SignTrans models and write metrics, reports, and confusion matrices."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_ml_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def evaluate_static(models_dir: str | Path, manifest_path: str | Path, split: str = "test") -> dict:
    from ml.dataset_loader import load_manifest
    from ml.evaluation import classification_metrics, save_evaluation
    from ml.model import load_keras_model
    from ml.models.artifacts import read_json
    from ml.training import build_static_dataset

    directory = Path(models_dir).expanduser().resolve()
    metadata = read_json(directory / "static_metadata.json")
    class_names = list(metadata["class_names"])
    manifest = load_manifest(manifest_path)
    dataset = build_static_dataset(
        manifest,
        split,
        class_names,
        image_size=tuple(metadata.get("image_size", [224, 224])),
        batch_size=32,
    )
    model = load_keras_model(directory / "static_model.keras")
    probabilities = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    metrics = classification_metrics(y_true, y_pred, class_names)
    save_evaluation(directory / "evaluation", metrics, class_names, prefix=f"static_{split}")
    logger.info("Static %s accuracy: %.4f", split, metrics["accuracy"])
    return metrics


def evaluate_dynamic(
    models_dir: str | Path,
    sequences_root: str | Path,
    *,
    split: str = "test",
    seed: int = 42,
) -> dict:
    from ml.evaluation import classification_metrics, save_evaluation
    from ml.model import load_keras_model
    from ml.models.artifacts import read_json
    from ml.training import _load_sequence_records, _sequence_array, _split_dynamic_records

    directory = Path(models_dir).expanduser().resolve()
    metadata = read_json(directory / "dynamic_metadata.json")
    class_names = list(metadata["class_names"])
    sequence_length = int(metadata.get("sequence_length", 30))
    records = _split_dynamic_records(_load_sequence_records(Path(sequences_root)), seed)
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise RuntimeError(f"No dynamic sequences were assigned to the {split} split")
    index = {label: position for position, label in enumerate(class_names)}
    unknown = sorted({record["label"] for record in selected} - set(index))
    if unknown:
        raise RuntimeError("Sequences contain labels absent from the model: " + ", ".join(unknown))
    features = np.stack([_sequence_array(Path(record["path"]), sequence_length) for record in selected])
    labels = np.asarray([index[record["label"]] for record in selected], dtype=np.int32)
    model = load_keras_model(directory / "dynamic_model.keras")
    probabilities = model.predict(features, verbose=0)
    metrics = classification_metrics(labels, np.argmax(probabilities, axis=1), class_names)
    save_evaluation(directory / "evaluation", metrics, class_names, prefix=f"dynamic_{split}")
    logger.info("Dynamic %s accuracy: %.4f", split, metrics["accuracy"])
    return metrics


def main() -> int:
    _ensure_ml_path()
    parser = argparse.ArgumentParser(description="Evaluate trained SignTrans models")
    sub = parser.add_subparsers(dest="command", required=True)

    static = sub.add_parser("static", help="Evaluate the static classifier")
    static.add_argument("--models-dir", default=str(Path(__file__).resolve().parent / "models"))
    static.add_argument("--manifest", required=True)
    static.add_argument("--split", default="test", choices=["train", "validation", "test"])

    dynamic = sub.add_parser("dynamic", help="Evaluate the dynamic LSTM classifier")
    dynamic.add_argument("--models-dir", default=str(Path(__file__).resolve().parent / "models"))
    dynamic.add_argument("--sequences", required=True)
    dynamic.add_argument("--split", default="test", choices=["train", "validation", "test"])
    dynamic.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "static":
        evaluate_static(args.models_dir, args.manifest, args.split)
    else:
        evaluate_dynamic(args.models_dir, args.sequences, split=args.split, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
