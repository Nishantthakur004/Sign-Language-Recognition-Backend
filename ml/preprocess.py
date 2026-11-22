"""Image validation and the single static preprocessing path used by ML code."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class InvalidImageError(ValueError):
    """Raised when input bytes or a file are not a safe, supported image."""


def _validate_image(image: Image.Image) -> Image.Image:
    if getattr(image, "is_animated", False):
        raise InvalidImageError("Animated images are not supported for static prediction")
    if image.format not in SUPPORTED_IMAGE_FORMATS:
        raise InvalidImageError(f"Unsupported image format: {image.format or 'unknown'}")
    width, height = image.size
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise InvalidImageError("Image dimensions are empty or exceed the safety limit")
    return ImageOps.exif_transpose(image).convert("RGB")


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    """Decode untrusted bytes fully and return an independent RGB Pillow image."""
    if not isinstance(image_bytes, bytes):
        raise InvalidImageError("image_bytes must be bytes")
    if not image_bytes:
        raise InvalidImageError("Image bytes are empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidImageError("Image exceeds the 20 MiB encoded-size limit")
    try:
        with Image.open(BytesIO(image_bytes)) as probe:
            if probe.format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidImageError(f"Unsupported image format: {probe.format or 'unknown'}")
            probe.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            return _validate_image(source)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("Could not decode a valid image") from exc


def load_image(path: Union[str, Path]) -> Image.Image:
    image_path = Path(path)
    try:
        if image_path.stat().st_size > MAX_IMAGE_BYTES:
            raise InvalidImageError("Image exceeds the 20 MiB encoded-size limit")
        return decode_image_bytes(image_path.read_bytes())
    except InvalidImageError:
        raise
    except OSError as exc:
        raise InvalidImageError(f"Could not read image: {image_path}") from exc


def preprocess_image(image: Image.Image, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize RGB input and apply MobileNetV2 scaling to [-1, 1]."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError("image_size must contain two positive dimensions")
    resized = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    return array / np.float32(127.5) - np.float32(1.0)


def preprocess_bytes(
    image_bytes: bytes, image_size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    return preprocess_image(decode_image_bytes(image_bytes), image_size)


def preprocess_path(path: Union[str, Path], image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    return preprocess_image(load_image(path), image_size)


def preprocess_batch(
    images: list[bytes], image_size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    if not images:
        raise InvalidImageError("At least one image is required")
    return np.stack([preprocess_bytes(item, image_size) for item in images]).astype(np.float32)
