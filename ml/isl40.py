"""ISL40 (vidit031/isl-isolated-40words) landmark extraction and word-model training.

The local release ships MP4 clips plus a small JSON sidecar of provenance per clip.
No landmark files are shipped, so this module extracts real MediaPipe two-hand
landmarks from the videos (cached to ``.h5``), builds a signer-aware split, and
trains the same word model the app already loads (``dynamic_model.keras``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.trainword import train_word_model
from ml.models.artifacts import utc_now, write_json
from ml.preprocessing.landmarks import TWO_HAND_FEATURE_DIM
from ml.video_inference import extract_video_frames

logger = logging.getLogger(__name__)

ISL40_ROOT = Path(__file__).resolve().parent / "datasets" / "ISL40"
ISL40_CACHE = ISL40_ROOT / ".cache" / "landmarks"
DATASET_NAME = "ISL40 (vidit031/isl-isolated-40words)"
DATASET_SOURCE = "https://huggingface.co/datasets/vidit031/isl-isolated-40words"
DEFAULT_SEQUENCE_LENGTH = 45
DEFAULT_MAX_FRAMES = 60
MAX_LANDMARK_FRAMES = 1800


class ISL40Error(RuntimeError):
    """Raised when the local ISL40 videos cannot produce a valid training set."""


def _word_dirs(root: Path = ISL40_ROOT) -> list[Path]:
    directories = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    return sorted(directories, key=lambda item: item.name.casefold())


def _signer_for(mp4: Path) -> str:
    sidecar = mp4.with_suffix(".json")
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return str(payload.get("signer") or payload.get("dataset") or "unknown").strip() or "unknown"
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return "unknown"


def prepare(force: bool = False, sample_every: int = 2) -> list[dict[str, Any]]:
    """Extract two-hand landmarks from every ISL40 clip and cache them as ``.h5``."""
    if not ISL40_ROOT.is_dir():
        raise ISL40Error(f"ISL40 dataset directory not found: {ISL40_ROOT}")
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ISL40Error("MediaPipe is required to prepare the ISL40 dataset") from exc

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    records: list[dict[str, Any]] = []
    try:
        for word_dir in _word_dirs():
            for mp4 in sorted(word_dir.glob("*.mp4"), key=lambda item: item.name.casefold()):
                destination = ISL40_CACHE / word_dir.name / f"{mp4.stem}.h5"
                if not destination.exists() or force:
                    try:
                        extracted = extract_video_frames(
                            mp4,
                            max_frames=MAX_LANDMARK_FRAMES,
                            sample_every=sample_every,
                            require_hands=False,
                            hands=hands,
                        )
                    except Exception as exc:  # noqa: BLE001 - skip unreadable clips
                        logger.warning("Skipping unreadable clip %s: %s", mp4.name, exc)
                        continue
                    frames = extracted["frames"]
                    if extracted["detected"] == 0 or frames is None or len(frames) == 0:
                        logger.warning("No hands detected in %s; skipping", mp4.name)
                        if destination.exists():
                            destination.unlink(missing_ok=True)
                        continue
                    if len(frames) > DEFAULT_MAX_FRAMES:
                        keep = np.clip(np.round(np.linspace(0, len(frames) - 1, DEFAULT_MAX_FRAMES)).astype(int), 0, len(frames) - 1)
                        frames = frames[keep]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    import h5py

                    with h5py.File(destination, "w") as handle:
                        handle.create_dataset("intermediate", data=frames.astype(np.float32))
                records.append(
                    {
                        "path": str(destination),
                        "label": word_dir.name,
                        "user": _signer_for(mp4),
                        "video": str(mp4),
                    }
                )
    finally:
        hands.close()
    if not records:
        raise ISL40Error("No ISL40 clips produced readable landmark sequences")
    logger.info("Prepared %d ISL40 landmark sequences into %s", len(records), ISL40_CACHE)
    return records


def _assign_splits(records: list[dict[str, Any]], seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-class stratified train/validation/test split.

    An earlier signer-holdout split reserved one signer per class for test, but
    many ISL40 signers contribute a single clip, so several words ended up with
    exactly one test sample. With a single sample a class's F1 is just 0% or
    100%, which made the per-class report noisy (whole words showing 0% purely
    because their one clip was misclassified). We now reserve a small, stable
    slice of every class for validation and test (at least two test clips once a
    class has enough data) while keeping the large majority for training, so the
    reported per-class F1 reflects several predictions instead of a coin flip.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)
    fallback_classes: list[str] = []
    for label in sorted(by_label):
        items = by_label[label]
        n = len(items)
        order = [items[index] for index in rng.permutation(n).tolist()]
        if n >= 8:
            test_n = max(2, int(round(n * 0.15)))
            val_n = max(1, int(round(n * 0.10)))
        elif n >= 5:
            test_n, val_n = 2, 1
        elif n == 4:
            test_n, val_n = 1, 1
        elif n == 3:
            test_n, val_n = 1, 0
        elif n == 2:
            test_n, val_n = 1, 0
        else:
            test_n, val_n = 0, 0
        # Keep at least two clips in the training split whenever possible.
        while n - test_n - val_n < 2 and (val_n > 0 or test_n > 1):
            if val_n > 0:
                val_n -= 1
            else:
                test_n -= 1
        for index, item in enumerate(order):
            if index < test_n:
                item["split"] = "test"
            elif index < test_n + val_n:
                item["split"] = "validation"
            else:
                item["split"] = "train"
        fallback_classes.append(label)
    strategy = {
        "strategy": "per-class stratified clip split",
        "test_signer_per_class": False,
        "validation_signer_per_class": False,
        "clip_level_fallback_classes": fallback_classes,
    }
    return records, strategy


def _split_signers(signers: list[str], seed: int = 42) -> dict[str, str]:
    """Kept for backward compatibility with older manifest readers."""
    ordered = sorted(set(signers))
    rng = np.random.default_rng(seed)
    shuffled = [ordered[index] for index in rng.permutation(len(ordered)).tolist()]
    assignment: dict[str, str] = {}
    for index, signer in enumerate(shuffled):
        fraction = index / max(1, len(shuffled))
        if fraction < 0.80:
            assignment[signer] = "train"
        elif fraction < 0.90:
            assignment[signer] = "validation"
        else:
            assignment[signer] = "test"
    return assignment


def build_manifest(
    output: str | Path | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    force_prepare: bool = False,
    sample_every: int = 2,
    seed: int = 42,
    min_train_per_class: int = 1,
) -> dict[str, Any]:
    records = prepare(force=force_prepare, sample_every=sample_every)
    _, split_strategy = _assign_splits(records, seed)

    classes = sorted({record["label"] for record in records})
    train_labels = {record["label"] for record in records if record["split"] == "train"}
    keep = [label for label in classes if sum(1 for r in records if r["label"] == label and r["split"] == "train") >= min_train_per_class]
    if len(keep) < 2:
        raise ISL40Error("Fewer than two classes have training sequences after the signer split")
    records = [record for record in records if record["label"] in keep]
    classes = keep
    records.sort(key=lambda item: (item["label"], item["path"]))

    distribution = Counter(record["label"] for record in records)
    split_counts = Counter(record["split"] for record in records)
    manifest = {
        "dataset": DATASET_NAME,
        "source": DATASET_SOURCE,
        "modality": "mediapipe_landmarks_from_video",
        "root": str(ISL40_ROOT),
        "sequence_length": sequence_length,
        "feature_dim": TWO_HAND_FEATURE_DIM,
        "classes": classes,
        "class_distribution": dict(sorted(distribution.items())),
        "users": sorted({record["user"] for record in records}),
        "split_strategy": split_strategy,
        "signer_split": split_strategy,
        "split_counts": dict(split_counts),
        "records": records,
        "samples": [{"label": label, "sample_image": None} for label in classes],
        "note": "Two-hand MediaPipe landmarks extracted from the local ISL40 MP4 clips; per-class signer-holdout split.",
        "created_at": utc_now(),
    }
    destination = (Path(output) if output else Path(__file__).resolve().parent / "models" / "isl40_manifest.json").expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, manifest)
    report = {
        "dataset": DATASET_NAME,
        "source": DATASET_SOURCE,
        "modality": manifest["modality"],
        "total_sequences": len(records),
        "total_classes": len(classes),
        "classes": classes,
        "class_distribution": manifest["class_distribution"],
        "split_strategy": split_strategy,
        "signer_split": split_strategy,
        "split_counts": dict(split_counts),
        "sequence_length": sequence_length,
        "feature_dim": TWO_HAND_FEATURE_DIM,
        "samples": manifest["samples"],
        "note": manifest["note"],
        "created_at": manifest["created_at"],
    }
    write_json(destination.parent / "dataset_report.json", report)
    return manifest


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    architecture: str = "bilstm",
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    augment_copies: int = 4,
    balance_per_class: int = 0,
    conv_frontend: bool = True,
    status_file: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the app's word model on the ISL40 manifest."""
    return train_word_model(
        manifest_path,
        output_dir,
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_per_class=None,
        augment_copies=augment_copies,
        balance_per_class=balance_per_class,
        conv_frontend=conv_frontend,
        status_file=status_file,
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, manifest, or train the ISL40 word model")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Extract MediaPipe landmarks from the ISL40 clips")
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--sample-every", type=int, default=2)

    man = sub.add_parser("manifest", help="Build the ISL40 manifest and dataset report")
    man.add_argument("--output", default=None)
    man.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    man.add_argument("--force", action="store_true")
    man.add_argument("--sample-every", type=int, default=2)
    man.add_argument("--seed", type=int, default=42)

    tr = sub.add_parser("train", help="Train the word model from the ISL40 manifest")
    tr.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "models" / "isl40_manifest.json"))
    tr.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "models"))
    tr.add_argument("--architecture", default="bilstm", choices=["lstm", "gru", "bilstm", "attention"])
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--learning-rate", type=float, default=1e-3)
    tr.add_argument("--augment-copies", type=int, default=2)
    tr.add_argument("--balance-per-class", type=int, default=0,
                    help="Oversample rare classes up to this many training examples")
    tr.add_argument("--seed", type=int, default=42, help="Random seed for training")
    tr.add_argument("--status-file", default=str(Path(__file__).resolve().parent / "models" / "training_status.json"))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "prepare":
        records = prepare(force=args.force, sample_every=args.sample_every)
        print(json.dumps({"clips": len(records)}, indent=2))
    elif args.command == "manifest":
        manifest = build_manifest(
            output=Path(args.output) if args.output else None,
            sequence_length=args.sequence_length,
            force_prepare=args.force,
            sample_every=args.sample_every,
            seed=args.seed,
        )
        print(json.dumps({"classes": len(manifest["classes"]), "records": len(manifest["records"])}, indent=2))
    else:
        metadata = train(
            args.manifest,
            args.output_dir,
            architecture=args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            augment_copies=args.augment_copies,
            balance_per_class=args.balance_per_class,
            seed=args.seed,
            status_file=args.status_file,
        )
        print(json.dumps({"model_version": metadata["model_version"], "test_metrics": metadata["test_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
