"""Command-line entry point for training the real SignTrans models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_ml_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    _ensure_ml_path()
    parser = argparse.ArgumentParser(description="Train the SignTrans static and/or dynamic models")
    sub = parser.add_subparsers(dest="command", required=True)

    static = sub.add_parser("static", help="Train the MobileNetV2 static-image classifier")
    static.add_argument("--manifest", required=True, help="Path to manifest.json")
    static.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "models"), help="Directory to write artifacts")
    static.add_argument("--epochs", type=int, default=12)
    static.add_argument("--fine-tune-epochs", type=int, default=4)
    static.add_argument("--batch-size", type=int, default=32)
    static.add_argument("--learning-rate", type=float, default=1e-3)
    static.add_argument("--seed", type=int, default=42)

    dynamic = sub.add_parser("dynamic", help="Train the LSTM temporal landmark classifier")
    dynamic.add_argument("--sequences", required=True, help="Directory of labelled videos/sequences")
    dynamic.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "models"), help="Directory to write artifacts")
    dynamic.add_argument("--sequence-length", type=int, default=30)
    dynamic.add_argument("--epochs", type=int, default=20)
    dynamic.add_argument("--batch-size", type=int, default=16)
    dynamic.add_argument("--learning-rate", type=float, default=1e-3)
    dynamic.add_argument("--seed", type=int, default=42)

    words = sub.add_parser("words", help="Train the word-level recurrent model from a landmark manifest")
    words.add_argument("--manifest", required=True, help="Landmark manifest (e.g. backend/ml/models/isl40_manifest.json)")
    words.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "models"), help="Directory to write artifacts")
    words.add_argument("--architecture", default="bilstm", choices=["lstm", "gru", "bilstm"])
    words.add_argument("--epochs", type=int, default=25)
    words.add_argument("--batch-size", type=int, default=32)
    words.add_argument("--learning-rate", type=float, default=1e-3)
    words.add_argument("--max-per-class", type=int, default=12)
    words.add_argument("--augment-copies", type=int, default=2, help="Augmented replicas of the training split")
    words.add_argument("--sequence-length", type=int, default=None, help="Override the manifest sequence length")
    words.add_argument("--status-file", default=None, help="JSON file updated with training progress")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "static":
        from ml.training import train_static_model

        metadata = train_static_model(
            args.manifest,
            args.output_dir,
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        logger.info(
            "Static training finished (model_version=%s, test_accuracy=%s)",
            metadata["model_version"],
            metadata["test_metrics"].get("accuracy"),
        )
    elif args.command == "dynamic":
        from ml.training import train_dynamic_model

        metadata = train_dynamic_model(
            args.sequences,
            args.output_dir,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        test_metrics = metadata.get("test_metrics") or {}
        logger.info(
            "Dynamic training finished (model_version=%s, test_accuracy=%s)",
            metadata["model_version"],
            test_metrics.get("accuracy"),
        )
    elif args.command == "words":
        from ml.trainword import train_word_model

        metadata = train_word_model(
            args.manifest,
            args.output_dir,
            architecture=args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_per_class=args.max_per_class,
            augment_copies=args.augment_copies,
            sequence_length=args.sequence_length,
            status_file=args.status_file,
        )
        test_metrics = metadata.get("test_metrics") or {}
        logger.info(
            "Word training finished (model_version=%s, test_accuracy=%s, top3=%s)",
            metadata["model_version"],
            test_metrics.get("accuracy"),
            test_metrics.get("top3_accuracy"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
