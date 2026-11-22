"""Application-facing inference for the real trained SignTrans models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml.labels import category_for_label
from ml.model import load_keras_model
from ml.models.artifacts import read_json
from ml.preprocess import InvalidImageError, preprocess_bytes
from ml.preprocessing.landmarks import (
    FEATURE_DIM,
    TWO_HAND_FEATURE_DIM,
    as_hand_frames,
    as_hand_pair_frames,
    pad_hand_pair_sequence,
    pad_sequence,
)


class ModelUnavailableError(RuntimeError):
    """Raised when no trained model artifact can be loaded."""


class InvalidSequenceError(ValueError):
    """Raised when temporal landmark data is malformed."""


_STATIC_MODEL = "static_model.keras"
_STATIC_META = "static_metadata.json"
_DYNAMIC_MODEL = "dynamic_model.keras"
_DYNAMIC_META = "dynamic_metadata.json"


def _top_k(probabilities: np.ndarray, class_names: list[str], k: int = 3) -> list[dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=float)
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [
        {
            "rank": position + 1,
            "sign": class_names[index],
            "confidence": round(float(probabilities[index]), 4),
            "category": category_for_label(class_names[index]),
        }
        for position, index in enumerate(top_indices)
    ]


class SignPredictor:
    """Load and run real Keras artifacts, never producing fabricated scores."""

    def __init__(
        self,
        models_dir: str | Path | None = None,
        *,
        static_only: bool = False,
        dynamic_only: bool = False,
        tta_copies: int | None = None,
    ) -> None:
        if models_dir is None:
            raise ModelUnavailableError("models_dir must point to a directory of trained models")
        self.models_dir = Path(models_dir).expanduser().resolve()
        if not self.models_dir.is_dir():
            raise ModelUnavailableError(f"Models directory does not exist: {self.models_dir}")

        self._static_model = None
        self._static_meta: dict[str, Any] = {}
        self._dynamic_model = None
        self._dynamic_meta: dict[str, Any] = {}
        self._hand: Any = None
        self._hand_unavailable = False

        static_path = self.models_dir / _STATIC_MODEL
        dynamic_path = self.models_dir / _DYNAMIC_MODEL

        if not static_only and static_path.exists() and (self.models_dir / _STATIC_META).exists():
            self._static_meta = read_json(self.models_dir / _STATIC_META)
            self._static_model = load_keras_model(static_path)
            self._validate_meta(self._static_meta, static_path)
        if not dynamic_only and dynamic_path.exists() and (self.models_dir / _DYNAMIC_META).exists():
            self._dynamic_meta = read_json(self.models_dir / _DYNAMIC_META)
            self._dynamic_model = load_keras_model(dynamic_path)
            self._validate_meta(self._dynamic_meta, dynamic_path)
        if tta_copies is None:
            tta_copies = int(self._dynamic_meta.get("tta_copies", 0) or 0)
        self._tta_copies = max(0, int(tta_copies))

        if self._static_model is None and self._dynamic_model is None:
            raise ModelUnavailableError(
                "No trained model found. Run backend/ml/train.py first, placing artifacts "
                f"({_STATIC_MODEL}/{_DYNAMIC_MODEL}) in {self.models_dir}."
            )

    @staticmethod
    def _validate_meta(meta: dict[str, Any], artifact: Path) -> None:
        if not meta.get("class_names"):
            raise ModelUnavailableError(f"Model metadata is missing class_names: {artifact}")

    def _get_hand(self):
        if self._hand is not None:
            return self._hand
        if self._hand_unavailable:
            return None
        try:
            from ml.preprocessing.landmarks import MediaPipeHandExtractor

            self._hand = MediaPipeHandExtractor(static_image_mode=True)
        except Exception:
            self._hand_unavailable = True
            self._hand = None
        return self._hand

    def close(self) -> None:
        if self._hand is not None:
            try:
                self._hand.close()
            except Exception:
                pass
            self._hand = None

    def predict(
        self,
        image_bytes: bytes,
        *,
        landmarks: Any = None,
        sequence: Any = None,
    ) -> dict[str, Any]:
        if sequence is not None:
            return self._predict_sequence(sequence)
        if landmarks is not None and self._dynamic_model is not None:
            return self._predict_sequence(landmarks)
        if self._static_model is not None:
            return self._predict_image(image_bytes)
        # Only a dynamic model is available; fall back to sequence handling.
        if landmarks is not None:
            return self._predict_sequence(landmarks)
        if sequence is not None:
            return self._predict_sequence(sequence)
        raise ModelUnavailableError("Dynamic-only build requires landmarks or a sequence")

    def _predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            input_array = preprocess_bytes(image_bytes, tuple(self._static_meta.get("image_size", [224, 224])))
        except InvalidImageError:
            raise
        input_array = np.expand_dims(input_array, axis=0)
        probabilities = self._static_model.predict(input_array, verbose=0)[0]
        class_names = list(self._static_meta["class_names"])
        argmax = int(np.argmax(probabilities))

        hand_detected: bool | None = None
        out_landmarks = None
        hand = self._get_hand()
        if hand is not None:
            from ml.preprocess import decode_image_bytes

            rgb = np.asarray(decode_image_bytes(image_bytes))
            out_landmarks = hand.extract(rgb)
            hand_detected = out_landmarks is not None
            out_landmarks = (
                [[round(float(v), 4) for v in point] for point in out_landmarks]
                if out_landmarks is not None
                else None
            )

        return {
            "sign": class_names[argmax],
            "confidence": round(float(probabilities[argmax]), 4),
            "category": self._static_meta.get("categories", {}).get(class_names[argmax]),
            "model_version": self._static_meta.get("model_version"),
            "recognition_type": "static",
            "top_k": _top_k(probabilities, class_names),
            "hand_detected": hand_detected,
            "landmarks": out_landmarks,
        }

    def _predict_sequence(self, sequence: Any) -> dict[str, Any]:
        if self._dynamic_model is None:
            raise ModelUnavailableError("Dynamic model is unavailable")
        sequence_length = int(self._dynamic_meta.get("sequence_length", 30))
        features = int(self._dynamic_meta.get("feature_dim", FEATURE_DIM))
        if features == TWO_HAND_FEATURE_DIM:
            frames = as_hand_pair_frames(sequence)
            input_array = pad_hand_pair_sequence(sequence, sequence_length).reshape(1, sequence_length, features)
        else:
            frames = as_hand_frames(sequence)
            input_array = pad_sequence(frames, sequence_length).reshape(1, sequence_length, features)
        if input_array.shape[2] != self._dynamic_model.input.shape[-1]:
            raise InvalidSequenceError(
                f"Landmark feature dimension {input_array.shape[2]} does not match the trained model"
            )
        probabilities = self._run_dynamic(input_array)[0]
        return self._format_dynamic(probabilities, bool(np.any(frames)))

    def predict_sequences(self, sequences: list[Any]) -> list[dict[str, Any]]:
        """Predict several landmark sequences in one batched forward pass.

        Used by the uploaded-video pipeline, where a clip is sliced into many
        windows. A single graph call is far faster than one ``.predict`` per
        window.
        """
        if self._dynamic_model is None:
            raise ModelUnavailableError("Dynamic model is unavailable")
        if not sequences:
            raise InvalidSequenceError("At least one sequence is required")
        sequence_length = int(self._dynamic_meta.get("sequence_length", 30))
        features = int(self._dynamic_meta.get("feature_dim", FEATURE_DIM))
        built: list[np.ndarray] = []
        hand_flags: list[bool] = []
        for sequence in sequences:
            if features == TWO_HAND_FEATURE_DIM:
                frames = as_hand_pair_frames(sequence)
                array = pad_hand_pair_sequence(sequence, sequence_length)
            else:
                frames = as_hand_frames(sequence)
                array = pad_sequence(frames, sequence_length)
            built.append(array.reshape(sequence_length, features))
            hand_flags.append(bool(np.any(frames)))
        batch = np.stack(built).astype(np.float32)
        if batch.shape[2] != self._dynamic_model.input.shape[-1]:
            raise InvalidSequenceError(
                f"Landmark feature dimension {batch.shape[2]} does not match the trained model"
            )
        probabilities = self._run_dynamic(batch)
        return [
            self._format_dynamic(probabilities[index], hand_flags[index])
            for index in range(len(built))
        ]

    def _run_dynamic(self, input_array: np.ndarray) -> np.ndarray:
        # Calling the model directly (a compiled graph call) is markedly faster
        # than ``model.predict`` for single/small batches.
        copies = int(getattr(self, "_tta_copies", 0) or 0)
        if copies <= 0:
            return np.asarray(self._dynamic_model(input_array, training=False))
        # Test-time augmentation: average the model over the original clip plus
        # several augmented views. This stabilises predictions for rare signs
        # whose held-out clips are few, and matches how the reported test
        # metrics were produced.
        from ml.trainword import _augment_padded

        views = [input_array]
        for view in range(copies):
            views.append(
                np.stack(
                    [_augment_padded(row, np.random.default_rng(view)) for row in input_array]
                ).astype(np.float32)
            )
        batch = np.concatenate(views, axis=0)
        probabilities = np.asarray(self._dynamic_model(batch, training=False), dtype=float)
        return probabilities.reshape(copies + 1, len(input_array), -1).mean(axis=0)

    def _format_dynamic(self, probabilities: np.ndarray, hand_detected: bool) -> dict[str, Any]:
        class_names = list(self._dynamic_meta["class_names"])
        argmax = int(np.argmax(probabilities))
        return {
            "sign": class_names[argmax],
            "confidence": round(float(probabilities[argmax]), 4),
            "category": self._dynamic_meta.get("categories", {}).get(class_names[argmax]),
            "model_version": self._dynamic_meta.get("model_version"),
            "recognition_type": "dynamic",
            "top_k": _top_k(probabilities, class_names),
            "hand_detected": hand_detected,
            "landmarks": None,
        }
