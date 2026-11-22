"""MediaPipe extraction and normalization for genuine temporal landmark data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

HAND_LANDMARKS = 21
COORDINATES = 3
HAND_COUNT = 2
FEATURE_DIM = HAND_LANDMARKS * COORDINATES
TWO_HAND_FEATURE_DIM = HAND_COUNT * FEATURE_DIM
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SEQUENCE_EXTENSIONS = {".npy", ".npz", ".csv", ".json", ".h5", ".hdf5"}


class InvalidSequenceError(ValueError):
    """Raised when temporal landmarks are malformed or contain no hand data."""


def as_hand_frames(sequence: Any) -> np.ndarray:
    """Convert a finite frame sequence to shape (frames, 21, 3)."""
    try:
        array = np.asarray(sequence, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidSequenceError("Landmarks must be a numeric sequence") from exc
    if array.ndim == 1:
        if array.size != FEATURE_DIM:
            raise InvalidSequenceError(f"A hand frame must contain {FEATURE_DIM} values")
        array = array.reshape(1, HAND_LANDMARKS, COORDINATES)
    elif array.ndim == 2 and array.shape[1] == FEATURE_DIM:
        array = array.reshape(-1, HAND_LANDMARKS, COORDINATES)
    elif array.ndim == 2 and array.shape[1:] == (HAND_LANDMARKS, COORDINATES):
        array = array.reshape(1, HAND_LANDMARKS, COORDINATES)
    elif array.ndim != 3 or array.shape[1:] != (HAND_LANDMARKS, COORDINATES):
        raise InvalidSequenceError(
            "Landmarks must have shape (frames, 21, 3) or (frames, 63)"
        )
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise InvalidSequenceError("Landmark sequences must be non-empty and finite")
    return array.astype(np.float32, copy=True)


def normalize_landmarks(sequence: Any) -> np.ndarray:
    """Normalize each hand frame relative to its wrist and maximum hand extent."""
    frames = as_hand_frames(sequence)
    for index, frame in enumerate(frames):
        if not np.any(frame):
            continue
        frame -= frame[0].copy()
        scale = float(np.linalg.norm(frame, axis=1).max())
        if scale > 1e-8:
            frame /= scale
    return frames


def pad_sequence(sequence: Any, sequence_length: int) -> np.ndarray:
    """Normalize, uniformly downsample long clips, and zero-pad to a fixed length."""
    if not isinstance(sequence_length, int) or sequence_length < 1:
        raise InvalidSequenceError("sequence_length must be a positive integer")
    frames = normalize_landmarks(sequence)
    if len(frames) > sequence_length:
        indices = np.linspace(0, len(frames) - 1, sequence_length).round().astype(int)
        frames = frames[indices]
    output = np.zeros((sequence_length, HAND_LANDMARKS, COORDINATES), dtype=np.float32)
    output[: len(frames)] = frames
    return output.reshape(sequence_length, FEATURE_DIM)


def as_hand_pair_frames(sequence: Any) -> np.ndarray:
    """Convert a two-hand frame sequence to shape (frames, 2, 21, 3)."""
    try:
        array = np.asarray(sequence, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidSequenceError("Landmarks must be a numeric sequence") from exc
    if array.ndim == 1:
        if array.size == TWO_HAND_FEATURE_DIM:
            array = array.reshape(1, HAND_COUNT, HAND_LANDMARKS, COORDINATES)
        elif array.size == FEATURE_DIM:
            array = array.reshape(1, HAND_LANDMARKS, COORDINATES)
        else:
            raise InvalidSequenceError(
                f"A hand frame must contain {FEATURE_DIM} or {TWO_HAND_FEATURE_DIM} values"
            )
    if array.ndim == 2 and array.shape[1] == TWO_HAND_FEATURE_DIM:
        array = array.reshape(-1, HAND_COUNT, HAND_LANDMARKS, COORDINATES)
    if array.ndim == 2 and array.shape[1] == FEATURE_DIM:
        array = array.reshape(-1, HAND_LANDMARKS, COORDINATES)
    if array.ndim == 3 and array.shape[1:] == (HAND_LANDMARKS, COORDINATES):
        padded = np.zeros((array.shape[0], HAND_COUNT, HAND_LANDMARKS, COORDINATES), dtype=np.float32)
        padded[:, 0] = array
        array = padded
    if array.ndim == 4 and array.shape[1:] == (HAND_COUNT, HAND_LANDMARKS, COORDINATES):
        pass
    else:
        raise InvalidSequenceError(
            "Two-hand landmarks must have shape (frames, 2, 21, 3), (frames, 126), or a single hand"
        )
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise InvalidSequenceError("Landmark sequences must be non-empty and finite")
    return array.astype(np.float32, copy=True)


def trim_hand_pair_frames(frames: np.ndarray) -> np.ndarray:
    """Drop leading and trailing frames where both hands are empty."""
    active = np.abs(frames).sum(axis=(1, 2, 3)) > 0
    if not active.any():
        return frames[0:1]
    first = int(np.argmax(active))
    last = len(active) - 1 - int(np.argmax(active[::-1]))
    return frames[first : last + 1]


def normalize_hand_pair(frames: np.ndarray) -> np.ndarray:
    """Wrist-relative, extent-scaled two-hand normalisation that keeps inter-hand geometry.

    The two hands are canonicalised by wrist x-order (left-most hand first) so the
    channel order is stable across the dataset and the browser.
    """
    result = frames.astype(np.float32, copy=True)
    for index, frame in enumerate(result):
        present = np.abs(frame).sum(axis=(1, 2)) > 0
        if not present.any():
            continue
        if present.all() and frame[0, 0, 0] > frame[1, 0, 0]:
            frame[[0, 1]] = frame[[1, 0]]
        elif present[0] and frame[0, 0, 0] >= 0.5:
            frame[[0, 1]] = frame[[1, 0]]
        elif present[1] and frame[1, 0, 0] < 0.5:
            frame[[0, 1]] = frame[[1, 0]]
        wrists = frame[present, 0]
        origin = wrists.mean(axis=0)
        frame[present] -= origin
        scale = float(np.linalg.norm(frame[present], axis=2).max())
        if scale > 1e-8:
            frame[present] /= scale
    return result


def pad_hand_pair_sequence(sequence: Any, sequence_length: int) -> np.ndarray:
    """Trim, normalise, uniformly downsample, and zero-pad two-hand landmarks to (length, 126)."""
    if not isinstance(sequence_length, int) or sequence_length < 1:
        raise InvalidSequenceError("sequence_length must be a positive integer")
    frames = as_hand_pair_frames(sequence)
    frames = trim_hand_pair_frames(frames)
    frames = normalize_hand_pair(frames)
    if len(frames) > sequence_length:
        indices = np.linspace(0, len(frames) - 1, sequence_length).round().astype(int)
        frames = frames[indices]
    output = np.zeros((sequence_length, HAND_COUNT, HAND_LANDMARKS, COORDINATES), dtype=np.float32)
    output[: len(frames)] = frames
    return output.reshape(sequence_length, TWO_HAND_FEATURE_DIM)


def load_h5_sequence(path: str | Path) -> np.ndarray:
    """Load a genuine pre-extracted two-hand ``.h5`` sequence, never a static image."""
    source = Path(path)
    try:
        import h5py
    except ImportError as exc:
        raise InvalidSequenceError("h5py is required to load .h5 landmark sequences") from exc
    try:
        with h5py.File(source, "r") as handle:
            if "intermediate" in handle:
                payload = handle["intermediate"][:]
            else:
                payload = handle[next(iter(handle.keys()))][:]
    except (OSError, KeyError, ValueError) as exc:
        raise InvalidSequenceError(f"Could not load landmark sequence: {source}") from exc
    return as_hand_pair_frames(payload)


def _sequence_from_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("landmarks", "sequence", "frames"):
            if key in payload:
                return payload[key]
        raise InvalidSequenceError(
            "JSON sequences must contain a landmarks, sequence, or frames field"
        )
    return payload


def load_sequence_file(path: str | Path) -> np.ndarray:
    """Load a genuine pre-extracted landmark sequence, never a static image."""
    source = Path(path)
    try:
        suffix = source.suffix.lower()
        if suffix == ".npy":
            payload = np.load(source, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                key = next(
                    (name for name in ("landmarks", "sequence", "frames") if name in archive),
                    None,
                )
                if key is None:
                    raise InvalidSequenceError(
                        "NPZ sequences must contain landmarks, sequence, or frames"
                    )
                payload = archive[key]
        elif suffix == ".json":
            payload = _sequence_from_json(json.loads(source.read_text(encoding="utf-8")))
        elif suffix == ".csv":
            with source.open(newline="", encoding="utf-8-sig") as handle:
                payload = [[float(value) for value in row] for row in csv.reader(handle) if row]
        elif suffix in {".h5", ".hdf5"}:
            # Two-hand cache files are already shaped (frames, 2, 21, 3).
            return load_h5_sequence(source)
        else:
            raise InvalidSequenceError(f"Unsupported landmark sequence format: {suffix}")
    except InvalidSequenceError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise InvalidSequenceError(f"Could not load landmark sequence: {source}") from exc
    return as_hand_frames(payload)


class MediaPipeHandExtractor:
    """Lazy MediaPipe Hands wrapper shared by video extraction and inference."""

    def __init__(self, static_image_mode: bool = False) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for hand landmark extraction; install requirements-ml.txt"
            ) from exc
        if not hasattr(mp, "solutions"):
            raise RuntimeError("This MediaPipe build does not provide the Hands solution")
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, rgb_frame: np.ndarray) -> np.ndarray | None:
        result = self._hands.process(np.asarray(rgb_frame, dtype=np.uint8))
        if not result.multi_hand_landmarks:
            return None
        return np.asarray(
            [[point.x, point.y, point.z] for point in result.multi_hand_landmarks[0].landmark],
            dtype=np.float32,
        )

    def draw(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        result = self._hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self._mp.solutions.drawing_utils.draw_landmarks(
                    bgr_frame, hand, self._mp.solutions.hands.HAND_CONNECTIONS
                )
        return bgr_frame

    def close(self) -> None:
        self._hands.close()


def extract_video_landmarks(
    path: str | Path, max_frames: int | None = None
) -> tuple[np.ndarray, list[bool]]:
    """Extract real MediaPipe landmarks from every frame of a labelled video."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video landmark extraction") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise InvalidSequenceError(f"Could not open video: {path}")
    extractor = MediaPipeHandExtractor(static_image_mode=False)
    frames: list[np.ndarray] = []
    detected: list[bool] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            landmarks = extractor.extract(frame[:, :, ::-1])
            detected.append(landmarks is not None)
            frames.append(
                np.zeros((HAND_LANDMARKS, COORDINATES), dtype=np.float32)
                if landmarks is None
                else landmarks
            )
    finally:
        capture.release()
        extractor.close()
    if not frames:
        raise InvalidSequenceError(f"Video has no decodable frames: {path}")
    if not any(detected):
        raise InvalidSequenceError(f"No hand landmarks detected in video: {path}")
    return np.stack(frames), detected


def iter_sequence_files(raw_dir: Path) -> Iterator[Path]:
    for path in sorted(raw_dir.rglob("*"), key=lambda item: str(item).casefold()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS | SEQUENCE_EXTENSIONS:
            yield path
