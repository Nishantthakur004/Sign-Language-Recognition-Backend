from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from ml.preprocess import InvalidImageError, preprocess_bytes, preprocess_image


def _jpeg_bytes(size=(120, 90), color=(30, 80, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_preprocess_bytes_returns_normalized_float_array():
    array = preprocess_bytes(_jpeg_bytes())
    assert array.shape == (224, 224, 3)
    assert array.dtype == np.float32
    assert array.min() >= -1.0 and array.max() <= 1.0


def test_preprocess_image_respects_custom_size():
    array = preprocess_image(Image.new("RGB", (48, 48), (10, 20, 30)), image_size=(112, 112))
    assert array.shape == (112, 112, 3)


def test_invalid_image_bytes_raise():
    with pytest.raises(InvalidImageError):
        preprocess_bytes(b"this is not an image")
    with pytest.raises(InvalidImageError):
        preprocess_bytes(b"")
