"""Dataset-agnostic training for word-level landmark models.

A manifest is a JSON document with ``classes``, ``records`` (each with ``path``,
``label`` and ``split``), ``sequence_length`` and ``feature_dim``. Both the ISL40
pipeline and the generic ``backend/ml/train.py words`` command use this module.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.evaluation import classification_metrics, save_evaluation
from ml.labels import category_for_label
from ml.model import build_sequence_model
from ml.models.artifacts import new_model_version, utc_now, write_json
from ml.preprocessing.landmarks import (
    COORDINATES,
    HAND_COUNT,
    HAND_LANDMARKS,
    TWO_HAND_FEATURE_DIM,
    load_h5_sequence,
    pad_hand_pair_sequence,
)

logger = logging.getLogger(__name__)
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_ARCHITECTURE = "bilstm"
DEFAULT_AUGMENT_COPIES = 2


class WordModelError(RuntimeError):
    """Raised when a manifest cannot produce a valid training set."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WordModelError(f"Could not read manifest: {source}") from exc
    if not isinstance(payload, dict) or not payload.get("records"):
        raise WordModelError(f"Manifest has no records: {source}")
    return payload


def _load_split(
    manifest: dict[str, Any],
    split: str,
    class_index: dict[str, int],
    *,
    sequence_length: int,
    max_per_class: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    per_class: Counter[str] = Counter()
    for record in manifest["records"]:
        if record.get("split") != split:
            continue
        label = record["label"]
        if max_per_class is not None and per_class[label] >= max_per_class:
            continue
        try:
            sequence = load_h5_sequence(record["path"])
            padded = pad_hand_pair_sequence(sequence, sequence_length)
        except Exception as exc:  # noqa: BLE001 - skip unreadable cached files
            logger.warning("Skipping unreadable landmark file %s: %s", record["path"], exc)
            continue
        arrays.append(padded)
        labels.append(class_index[label])
        per_class[label] += 1
    if not arrays:
        return np.empty((0, sequence_length, TWO_HAND_FEATURE_DIM), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(arrays).astype(np.float32), np.asarray(labels, dtype=np.int32)


def _augment_padded(padded: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augment one normalized ``(sequence_length, 126)`` sample.

    The transforms target signer variation (which is what breaks generalisation
    to unseen people) rather than arbitrary noise: mirroring, small in-plane
    rotation, scale jitter, time-warping and landmark jitter. Only the non-empty
    (un-padded) frames are touched so the masking stays valid.
    """
    frames = np.asarray(padded, dtype=np.float32).reshape(
        -1, HAND_COUNT, HAND_LANDMARKS, COORDINATES
    ).copy()
    active = np.abs(frames).sum(axis=(1, 2, 3)) > 0
    if not active.any():
        return np.asarray(padded, dtype=np.float32)
    sequence = frames[active].copy()
    length = len(sequence)

    if rng.random() < 0.5:  # mirror across the vertical axis: swap hands, flip x
        sequence = sequence[:, ::-1, :, :].copy()
        sequence[..., 0] *= -1.0

    angle = float(rng.uniform(-0.18, 0.18))  # in-plane rotation in radians
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    x = sequence[..., 0].copy()
    y = sequence[..., 1].copy()
    sequence[..., 0] = cos * x - sin * y
    sequence[..., 1] = sin * x + cos * y

    sequence *= float(rng.uniform(0.9, 1.1))  # scale jitter

    if length >= 3 and rng.random() < 0.7:  # time warp, resampled to same length
        # A wide speed range makes the model tolerant of faster/slower signing.
        speed = float(rng.uniform(0.7, 1.4))
        target = max(2, int(round(length * speed)))
        source = np.linspace(0, length - 1, target)
        warped = sequence[np.clip(np.round(source).astype(int), 0, length - 1)]
        destination = np.linspace(0, target - 1, length)
        sequence = warped[np.clip(np.round(destination).astype(int), 0, target - 1)]

    sequence = sequence + rng.normal(0.0, 0.008, size=sequence.shape).astype(np.float32)
    frames[active] = sequence
    return frames.reshape(-1, TWO_HAND_FEATURE_DIM).astype(np.float32)


def _expand_with_augmentation(
    features: np.ndarray,
    labels: np.ndarray,
    copies: int,
    rng: np.random.Generator,
    copies_by_label: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the original split plus augmented replicas.

    When ``copies_by_label`` is given, each class receives its own number of
    augmented replicas (used to oversample rare words), otherwise every class
    gets ``copies`` replicas.
    """
    if len(features) == 0:
        return features, labels
    batches_x = [features]
    batches_y = [labels]
    if copies_by_label is None:
        if copies <= 0:
            return features, labels
        for _ in range(copies):
            batches_x.append(np.stack([_augment_padded(sample, rng) for sample in features]))
            batches_y.append(labels)
        return np.concatenate(batches_x, axis=0), np.concatenate(batches_y, axis=0)

    max_copies = int(copies_by_label.max()) if len(copies_by_label) else 0
    for round_index in range(max_copies):
        extra_x: list[np.ndarray] = []
        extra_y: list[int] = []
        for sample, label in zip(features, labels):
            if round_index < int(copies_by_label[int(label)]):
                extra_x.append(_augment_padded(sample, rng))
                extra_y.append(int(label))
        if extra_x:
            batches_x.append(np.stack(extra_x))
            batches_y.append(np.asarray(extra_y, dtype=labels.dtype))
    return np.concatenate(batches_x, axis=0), np.concatenate(batches_y, axis=0)


def _top_k_accuracy(probabilities: np.ndarray, truth: np.ndarray, k: int = 3) -> float:
    if len(truth) == 0:
        return 0.0
    top = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([truth[index] in top[index] for index in range(len(truth))]))


def _class_sample_weights(
    labels: np.ndarray, num_classes: int, cap: float = 8.0
) -> np.ndarray:
    """Inverse-frequency sample weights so rare words are not drowned out.

    The ISL40 classes are heavily imbalanced (one word has a single clip while
    others have ~40), so balancing the loss is one of the largest single wins for
    unseen-signer accuracy. Weights are capped so a 1-sample class cannot
    dominate the gradient.
    """
    if len(labels) == 0:
        return np.empty((0,), dtype=np.float32)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    if present.any():
        balanced = len(labels) / (present.sum() * counts[present])
        weights[present] = np.minimum(balanced, cap)
    return weights[labels].astype(np.float32)


def _status_writer(path: Path | None):
    def write(payload: dict[str, Any]) -> None:
        if path is None:
            return
        write_json(path.expanduser().resolve(), payload)

    return write


def train_word_model(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    architecture: str = DEFAULT_ARCHITECTURE,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    sequence_length: int | None = None,
    max_per_class: int | None = 12,
    augment_copies: int = DEFAULT_AUGMENT_COPIES,
    balance_per_class: int = 0,
    label_smoothing: float = 0.05,
    conv_frontend: bool = False,
    use_class_weights: bool = True,
    seed: int = 42,
    status_file: str | Path | None = None,
) -> dict[str, Any]:
    """Train a real recurrent word classifier from a landmark manifest."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise WordModelError(
            "TensorFlow is required for training. Install backend/requirements.txt first."
        ) from exc

    manifest = load_manifest(manifest_path)
    class_names = list(manifest["classes"])
    index = {label: position for position, label in enumerate(class_names)}
    length = int(sequence_length or manifest.get("sequence_length") or DEFAULT_SEQUENCE_LENGTH)
    if max_per_class is not None and max_per_class <= 0:
        max_per_class = None
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    status = _status_writer(Path(status_file) if status_file else None)
    started = utc_now()
    status(
        {
            "state": "preparing",
            "started_at": started,
            "architecture": architecture,
            "classes": len(class_names),
            "epochs": epochs,
        }
    )

    tf.keras.utils.set_random_seed(seed)
    train_x, train_y = _load_split(
        manifest, "train", index, sequence_length=length, max_per_class=max_per_class
    )
    validation_x, validation_y = _load_split(
        manifest, "validation", index, sequence_length=length, max_per_class=max_per_class
    )
    test_x, test_y = _load_split(
        manifest, "test", index, sequence_length=length, max_per_class=max_per_class
    )
    if len(train_x) == 0:
        raise WordModelError("The training split contains no readable landmark sequences")
    if len(validation_x) == 0:
        logger.warning("Validation split is empty; reusing a slice of the training split")
        validation_x, validation_y = train_x[: max(1, len(train_x) // 10)], train_y[: max(1, len(train_x) // 10)]

    # Augment only the training split; validation and test stay untouched so the
    # reported metrics remain honest measurements on real held-out signers.
    rng = np.random.default_rng(seed)
    real_train_count = len(train_x)
    if balance_per_class and balance_per_class > 0:
        counts = np.bincount(train_y, minlength=len(class_names)).astype(np.float64)
        copies_by_label = np.zeros(len(class_names), dtype=int)
        present = counts > 0
        copies_by_label[present] = np.maximum(
            0, np.ceil(balance_per_class / counts[present]).astype(int) - 1
        )
        copies_by_label = np.minimum(copies_by_label, 40)
        train_x, train_y = _expand_with_augmentation(
            train_x, train_y, augment_copies, rng, copies_by_label=copies_by_label
        )
    else:
        train_x, train_y = _expand_with_augmentation(train_x, train_y, augment_copies, rng)
    sample_weight = (
        _class_sample_weights(train_y, len(class_names)) if use_class_weights else None
    )
    logger.info(
        "Training on %d sequences (%d real + %d augmented)%s",
        len(train_x),
        real_train_count,
        len(train_x) - real_train_count,
        " with class-balanced sample weights" if use_class_weights else "",
    )

    status(
        {
            "state": "training",
            "started_at": started,
            "architecture": architecture,
            "classes": len(class_names),
            "epochs": epochs,
            "samples": {"train": int(len(train_x)), "validation": int(len(validation_x)), "test": int(len(test_x))},
        }
    )

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            status(
                {
                    "state": "training",
                    "started_at": started,
                    "architecture": architecture,
                    "classes": len(class_names),
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "accuracy": float(logs.get("accuracy", 0.0)),
                    "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                    "loss": float(logs.get("loss", 0.0)),
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "samples": {
                        "train": int(len(train_x)),
                        "validation": int(len(validation_x)),
                        "test": int(len(test_x)),
                    },
                }
            )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(destination / "dynamic_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
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
        tf.keras.callbacks.CSVLogger(str(destination / "training_log.csv")),
        Progress(),
    ]

    model = build_sequence_model(
        class_names,
        length,
        feature_dim=TWO_HAND_FEATURE_DIM,
        learning_rate=learning_rate,
        architecture=architecture,
        label_smoothing=label_smoothing,
        conv_frontend=conv_frontend,
    )
    history = model.fit(
        train_x,
        train_y,
        sample_weight=sample_weight,
        validation_data=(validation_x, validation_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    model_path = destination / "dynamic_model.keras"
    model.save(model_path)

    metrics: dict[str, Any] = {}
    if len(test_x):
        probabilities = model.predict(test_x, verbose=0)
        metrics = classification_metrics(test_y, np.argmax(probabilities, axis=1), class_names)
        metrics["top3_accuracy"] = round(_top_k_accuracy(probabilities, test_y, 3), 6)
        save_evaluation(destination / "evaluation", metrics, class_names, prefix="dynamic")

    version = new_model_version("words")
    metadata = {
        "model_version": version,
        "recognition_type": "dynamic",
        "model_file": model_path.name,
        "architecture": architecture,
        "class_names": class_names,
        "categories": {label: category_for_label(label) for label in class_names},
        "sequence_length": length,
        "feature_dim": TWO_HAND_FEATURE_DIM,
        "dataset": manifest.get("dataset"),
        "dataset_source": manifest.get("source"),
        "dataset_manifest": str(Path(manifest_path).resolve()),
        "dataset_root": manifest.get("root"),
        "signer_split": manifest.get("signer_split"),
        "training": {
            "epochs": len(history.history.get("loss", [])),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_per_class": max_per_class,
            "augment_copies": augment_copies,
            "balance_per_class": balance_per_class,
            "label_smoothing": label_smoothing,
            "conv_frontend": conv_frontend,
            "class_weights": use_class_weights,
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        },
        "test_metrics": metrics,
        "created_at": utc_now(),
    }
    write_json(destination / "dynamic_metadata.json", metadata)
    write_json(
        destination / "label_mapping.json",
        {"classes": class_names, "index": {label: position for position, label in enumerate(class_names)}},
    )
    write_json(
        destination / "dataset_report.json",
        {
            "dataset": manifest.get("dataset"),
            "source": manifest.get("source"),
            "modality": manifest.get("modality"),
            "total_sequences": len(manifest["records"]),
            "total_classes": len(class_names),
            "classes": class_names,
            "class_distribution": manifest.get("class_distribution", {}),
            "users": manifest.get("users", []),
            "signer_split": manifest.get("signer_split", {}),
            "split_counts": manifest.get("split_counts", {}),
            "samples": [{"label": label, "sample_image": None} for label in class_names],
            "note": manifest.get("note") or "Word-level MediaPipe landmarks only.",
            "created_at": utc_now(),
        },
    )
    status(
        {
            "state": "completed",
            "started_at": started,
            "finished_at": utc_now(),
            "architecture": architecture,
            "classes": len(class_names),
            "model_version": version,
            "test_metrics": metrics,
            "history": metadata["training"]["history"],
        }
    )
    logger.info("Word model saved to %s (test accuracy %.4f)", model_path, metrics.get("accuracy", 0.0))
    return metadata
