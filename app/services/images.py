from __future__ import annotations

import base64
import binascii
import io
import re
import warnings

from PIL import Image, UnidentifiedImageError

DATA_URL_RE = re.compile(r"^data:(image/(?:jpeg|png));base64,([A-Za-z0-9+/=\r\n]+)$", re.IGNORECASE)


class ImageValidationError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_image", status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def validate_image_bytes(data: bytes, max_bytes: int, declared_type: str | None = None) -> str:
    if not data:
        raise ImageValidationError("Image is empty")
    if len(data) > max_bytes:
        raise ImageValidationError("Image exceeds the upload size limit", code="image_too_large", status_code=413)
    if declared_type is not None and declared_type.lower() not in {"image/jpeg", "image/png"}:
        raise ImageValidationError("Only JPEG and PNG images are supported", code="unsupported_media_type", status_code=415)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format
                image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("Invalid or corrupt JPEG/PNG image") from exc
    if image_format not in {"JPEG", "PNG"}:
        raise ImageValidationError("Only JPEG and PNG images are supported", code="unsupported_media_type", status_code=415)
    expected = "JPEG" if declared_type and declared_type.lower() == "image/jpeg" else "PNG" if declared_type else image_format
    if declared_type and image_format != expected:
        raise ImageValidationError("Image content does not match its declared media type")
    return "image/jpeg" if image_format == "JPEG" else "image/png"


def decode_data_url(value: str, max_bytes: int) -> bytes:
    match = DATA_URL_RE.fullmatch(value)
    if match is None:
        raise ImageValidationError("Frame must be a base64 JPEG or PNG data URL", code="invalid_frame")
    encoded = "".join(match.group(2).split())
    if len(encoded) > ((max_bytes + 2) // 3) * 4 + 4:
        raise ImageValidationError("Image exceeds the upload size limit", code="image_too_large", status_code=413)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageValidationError("Frame contains invalid base64 data", code="invalid_frame") from exc
    validate_image_bytes(data, max_bytes, match.group(1).lower())
    return data
