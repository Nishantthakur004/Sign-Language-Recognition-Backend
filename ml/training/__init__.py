"""TensorFlow data pipelines and training routines for static and dynamic ISL signs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml.dataset_loader import DatasetError, load_manifest, records_for_split, resolve_record_path
from ml.evaluation import classification_metrics, save_evaluation
from ml.labels import category_for_label
from ml.model import build_dynamic_model, build_static_model
from ml.models.artifacts import new_model_version, utc_now, write_json
from ml.preprocess import IMAGE_SIZE
from ml.preprocessing.landmarks import (
    FEATURE_DIM,
    extract_video_landmarks,
    load_sequence_file,
    pad_sequence,
)

logger = logging.getLogger(__name__)


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install backend/requirements.txt first."
        ) from exc
    return tf


def _decode_image(path: Any, label: Any, image_size: tuple[int, int]):
    tf = _tf()
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image, label


def build_static_dataset(
    manifest: dict[str, Any],
    split: str,
    class_names: Sequence[str],
    *,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
):
    """Build a tf.data pipeline from the real manifest; no synthetic labels are added."""
    tf = _tf()
    records = records_for_split(manifest, split)
    index = {label: position for position, label in enumerate(class_names)}
    paths = [str(resolve_record_path(manifest, record)) for record in records]
    labels = [index[record["label"]] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(min(len(paths), 10_000), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _decode_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _callbacks(tf, output_dir: Path, checkpoint_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / checkpoint_name),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def _evaluate_dataset(model, dataset, class_names: Sequence[str]) -> dict[str, Any]:
    probabilities = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    return classification_metrics(y_true, y_pred, class_names)


def train_static_model(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 12,
    fine_tune_epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train MobileNetV2 transfer learning on the real dataset manifest."""
    tf = _tf()
    tf.keras.utils.set_random_seed(seed)
    manifest = load_manifest(manifest_path)
    class_names = list(manifest["classes"])
    if len(class_names) < 2:
        raise DatasetError("At least two classes are required for training")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    train_dataset = build_static_dataset(
        manifest, "train", class_names, batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_dataset = build_static_dataset(
        manifest, "validation", class_names, batch_size=batch_size
    )
    test_dataset = build_static_dataset(manifest, "test", class_names, batch_size=batch_size)

    model = build_static_model(class_names, learning_rate=learning_rate)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=_callbacks(tf, destination, "static_model.keras"),
        verbose=1,
    )

    if fine_tune_epochs > 0:
        base = next(
            (layer for layer in model.layers if layer.name.startswith("mobilenetv2")),
            None,
        )
        if base is not None:
            base.trainable = True
            for layer in base.layers[:-30]:
                layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 10),
                loss="sparse_categorical_crossentropy",
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )
            fine_history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                initial_epoch=len(history.history["loss"]),
                epochs=len(history.history["loss"]) + fine_tune_epochs,
                callbacks=_callbacks(tf, destination, "static_model.keras"),
                verbose=1,
            )
            for key, values in fine_history.history.items():
                history.history.setdefault(key, []).extend(values)

    model_path = destination / "static_model.keras"
    model.save(model_path)
    metrics = _evaluate_dataset(model, test_dataset, class_names)
    version = new_model_version("static")
    metadata = {
        "model_version": version,
        "recognition_type": "static",
        "model_file": model_path.name,
        "class_names": class_names,
        "categories": {label: category_for_label(label) for label in class_names},
        "image_size": list(IMAGE_SIZE),
        "dataset_manifest": str(Path(manifest_path).resolve()),
        "dataset_root": manifest.get("root"),
        "training": {
            "epochs": len(history.history.get("loss", [])),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "fine_tune_epochs": fine_tune_epochs,
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        },
        "test_metrics": metrics,
        "created_at": utc_now(),
    }
    write_json(destination / "static_metadata.json", metadata)
    save_evaluation(destination / "evaluation", metrics, class_names, prefix="static")
    logger.info("Static model saved to %s with test accuracy %.4f", model_path, metrics["accuracy"])
    return metadata


def _load_sequence_records(records_root: Path) -> list[dict[str, Any]]:
    """Discover labelled videos or pre-extracted landmark sequences."""
    from ml.preprocessing.landmarks import SEQUENCE_EXTENSIONS, VIDEO_EXTENSIONS

    records: list[dict[str, Any]] = []
    for path in sorted(records_root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS | SEQUENCE_EXTENSIONS:
            continue
        relative = path.relative_to(records_root)
        if len(relative.parts) < 2:
            raise DatasetError(f"Dynamic sequence is not inside a label directory: {path}")
        label = relative.parts[-2]
        group = "/".join(relative.parts[:-2]) or "default"
        records.append({"path": str(path), "label": label, "group": group})
    if not records:
        raise DatasetError(
            "No dynamic videos or landmark sequences were found. Dynamic training requires "
            "labelled temporal data; static images are never used for the LSTM."
        )
    return records


def _split_dynamic_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    from ml.dataset_loader import ImageRecord, split_records

    pseudo = [
        ImageRecord(path=record["path"], label=record["label"], group=record["group"], split="")
        for record in records
    ]
    split, _ = split_records(pseudo, seed=seed, allow_ungrouped=True)
    mapping = {item.path: item.split for item in split}
    return [{**record, "split": mapping[record["path"]]} for record in records]


def _sequence_array(path: Path, sequence_length: int) -> np.ndarray:
    from ml.preprocessing.landmarks import VIDEO_EXTENSIONS

    if path.suffix.lower() in VIDEO_EXTENSIONS:
        frames, _ = extract_video_landmarks(path)
        return pad_sequence(frames, sequence_length)
    return pad_sequence(load_sequence_file(path), sequence_length)


def train_dynamic_model(
    sequences_root: str | Path,
    output_dir: str | Path,
    *,
    sequence_length: int = 30,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train an LSTM only from labelled videos or genuine landmark sequences."""
    tf = _tf()
    tf.keras.utils.set_random_seed(seed)
    root = Path(sequences_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(f"Dynamic sequence directory does not exist: {root}")
    records = _split_dynamic_records(_load_sequence_records(root), seed)
    class_names = sorted({record["label"] for record in records})
    if len(class_names) < 2:
        raise DatasetError("At least two dynamic classes are required")
    index = {label: position for position, label in enumerate(class_names)}

    arrays: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    labels: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for record in records:
        sequence = _sequence_array(Path(record["path"]), sequence_length)
        arrays[record["split"]].append(sequence)
        labels[record["split"]].append(index[record["label"]])
    if not arrays["train"]:
        raise DatasetError("Dynamic split produced no training sequences")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    train_x = np.stack(arrays["train"]).astype(np.float32)
    train_y = np.asarray(labels["train"], dtype=np.int32)
    validation_x = np.stack(arrays["validation"]).astype(np.float32) if arrays["validation"] else train_x[:1]
    validation_y = np.asarray(labels["validation"] or labels["train"][:1], dtype=np.int32)

    model = build_dynamic_model(class_names, sequence_length, feature_dim=FEATURE_DIM, learning_rate=learning_rate)
    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=_callbacks(tf, destination, "dynamic_model.keras"),
        verbose=1,
    )
    model_path = destination / "dynamic_model.keras"
    model.save(model_path)

    metrics: dict[str, Any] = {}
    if arrays["test"]:
        test_x = np.stack(arrays["test"]).astype(np.float32)
        test_y = np.asarray(labels["test"], dtype=np.int32)
        probabilities = model.predict(test_x, verbose=0)
        metrics = classification_metrics(test_y, np.argmax(probabilities, axis=1), class_names)
        save_evaluation(destination / "evaluation", metrics, class_names, prefix="dynamic")

    version = new_model_version("dynamic")
    metadata = {
        "model_version": version,
        "recognition_type": "dynamic",
        "model_file": model_path.name,
        "class_names": class_names,
        "categories": {label: category_for_label(label) for label in class_names},
        "sequence_length": sequence_length,
        "feature_dim": FEATURE_DIM,
        "dataset_root": str(root),
        "training": {
            "epochs": len(history.history.get("loss", [])),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        },
        "test_metrics": metrics,
        "created_at": utc_now(),
    }
    write_json(destination / "dynamic_metadata.json", metadata)
    logger.info("Dynamic model saved to %s", model_path)
    return metadata
