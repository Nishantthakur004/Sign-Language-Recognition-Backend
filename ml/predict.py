"""Command-line prediction against a trained SignTrans model."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Run a real prediction with a trained SignTrans model")
    parser.add_argument("--models-dir", default=str(Path(__file__).resolve().parent / "models"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a static gesture image")
    source.add_argument("--sequence", help="Path to a landmark sequence (.h5/.npy/.npz/.csv/.json)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from ml.inference.predictor import ModelUnavailableError, SignPredictor
    from ml.preprocessing.landmarks import load_sequence_file

    try:
        predictor = SignPredictor(models_dir=args.models_dir)
    except ModelUnavailableError as exc:
        logger.error("%s", exc)
        return 2

    try:
        if args.image:
            result = predictor.predict(Path(args.image).read_bytes())
        else:
            result = predictor.predict(b"", sequence=load_sequence_file(args.sequence))
    except (OSError, ValueError) as exc:
        logger.error("Prediction failed: %s", exc)
        return 1
    finally:
        predictor.close()

    result["top_k"] = result.get("top_k", [])[: max(1, args.top_k)]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
