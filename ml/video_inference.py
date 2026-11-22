"""Uploaded-video recognition from real MediaPipe landmarks.

The uploaded clip is decoded with OpenCV, hand landmarks are extracted with
MediaPipe (two hands, ordered the same way as the training data), sliced into
overlapping windows, and classified by the trained word model. Nothing here
fabricates predictions: when no hands are found the caller gets an explicit
error instead of a made-up sign.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

HAND_COUNT = 2
HAND_LANDMARKS = 21
COORDINATES = 3
DEFAULT_WINDOW = 30
DEFAULT_STRIDE = 15
# ISL40 clips are ~2 s long, so recognition windows match that duration; a
# 4 s window would feed the model twice the temporal context it was trained on.
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_STRIDE_SECONDS = 1.0
MAX_VIDEO_FRAMES = 1800
DEFAULT_MIN_CONFIDENCE = 0.3


class VideoRecognitionError(ValueError):
    """Raised when an uploaded video cannot be decoded or contains no hands."""


def _ordered_frame(hands: list[np.ndarray]) -> np.ndarray:
    """Build a (2, 21, 3) frame with the same hand ordering used in training."""
    frame = np.zeros((HAND_COUNT, HAND_LANDMARKS, COORDINATES), dtype=np.float32)
    if not hands:
        return frame
    ordered = sorted(hands, key=lambda hand: float(hand[0][0]))
    if len(ordered) == 1:
        index = 0 if float(ordered[0][0][0]) < 0.5 else 1
        frame[index] = ordered[0]
        return frame
    frame[0] = ordered[0]
    frame[1] = ordered[1]
    return frame


def extract_video_frames(
    path: str | Path,
    *,
    max_frames: int = MAX_VIDEO_FRAMES,
    sample_every: int = 1,
    require_hands: bool = True,
    max_width: int = 640,
    hands: Any = None,
) -> dict[str, Any]:
    """Decode a video and return real two-hand landmark frames plus metadata.

    Frames are downscaled to ``max_width`` before MediaPipe runs: landmark
    coordinates are normalised, so this is much faster with no accuracy loss.
    ``require_hands=False`` is used by dataset preparation, where a clip with no
    detected hands is simply skipped instead of failing the whole run. Passing an
    existing ``hands`` instance avoids re-initialising MediaPipe per clip.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise VideoRecognitionError(
            "Video recognition requires OpenCV and MediaPipe (install backend/requirements.txt)"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoRecognitionError("The uploaded video could not be opened")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 30.0

    owns_hands = hands is None
    if hands is None:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=HAND_COUNT,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    frames: list[np.ndarray] = []
    detected = 0
    try:
        index = 0
        while len(frames) < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            if sample_every > 1 and index % sample_every:
                index += 1
                continue
            index += 1
            if max_width and bgr.shape[1] > max_width:
                scale = max_width / float(bgr.shape[1])
                bgr = cv2.resize(
                    bgr,
                    (max_width, max(1, int(round(bgr.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            hand_arrays: list[np.ndarray] = []
            if result.multi_hand_landmarks:
                for landmark in result.multi_hand_landmarks:
                    hand_arrays.append(
                        np.asarray(
                            [[point.x, point.y, point.z] for point in landmark.landmark],
                            dtype=np.float32,
                        )
                    )
            if hand_arrays:
                detected += 1
            frames.append(_ordered_frame(hand_arrays))
    finally:
        capture.release()
        if owns_hands:
            hands.close()

    if not frames:
        raise VideoRecognitionError("The uploaded video has no decodable frames")
    if detected == 0 and require_hands:
        raise VideoRecognitionError("No hands were detected in the uploaded video")
    return {
        "frames": np.stack(frames),
        "fps": fps,
        "detected": detected,
        "sampled_frames": len(frames),
    }


def build_windows(
    frames: np.ndarray,
    fps: float,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Slice frames into overlapping windows sized in seconds (frame-rate aware).

    The model was trained on ~2 s ISL40 clips resampled to a fixed number of
    steps. Sizing windows by time rather than a raw frame count keeps that
    temporal scale regardless of the uploaded video's frame rate.
    """
    total = len(frames)
    if total == 0:
        return [], []
    effective_fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
    window = max(DEFAULT_WINDOW, int(round(effective_fps * window_seconds)))
    stride = max(1, int(round(effective_fps * stride_seconds)))
    if total <= window:
        return [frames], [(0, total)]
    windows: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    start = 0
    while start + window <= total:
        windows.append(frames[start : start + window])
        spans.append((start, start + window))
        start += stride
    if spans[-1][1] < total:
        windows.append(frames[total - window : total])
        spans.append((total - window, total))
    return windows, spans


def summarize_predictions(
    predictions: list[dict[str, Any]],
    spans: list[tuple[int, int]],
    *,
    fps: float,
    total_frames: int,
    detected_frames: int,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Collapse window predictions into a readable sign sequence and text."""
    segments: list[dict[str, Any]] = []
    for (start, end), prediction in zip(spans, predictions):
        if not prediction.get("hand_detected"):
            continue
        confidence = float(prediction.get("confidence") or 0.0)
        if confidence < min_confidence:
            continue
        segments.append(
            {
                "sign": prediction["sign"],
                "confidence": round(confidence, 4),
                "start_seconds": round(start / fps, 2),
                "end_seconds": round(end / fps, 2),
                "top_k": list(prediction.get("top_k") or [])[:3],
            }
        )

    merged: list[dict[str, Any]] = []
    for segment in segments:
        if merged and merged[-1]["sign"] == segment["sign"]:
            if segment["confidence"] > merged[-1]["confidence"]:
                merged[-1]["confidence"] = segment["confidence"]
                merged[-1]["top_k"] = segment["top_k"]
            merged[-1]["end_seconds"] = segment["end_seconds"]
            continue
        merged.append(dict(segment))

    return {
        "text": " ".join(item["sign"] for item in merged),
        "segments": merged,
        "model_version": predictions[0].get("model_version") if predictions else None,
        "fps": round(fps, 2),
        "total_frames": int(total_frames),
        "hands_detected_frames": int(detected_frames),
        "duration_seconds": round(total_frames / fps, 2),
    }
