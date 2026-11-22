from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import secrets
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.config import BACKEND_DIR, Settings
from app.schemas.recognition import PredictionResult

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class PredictionService:
    """Lazy adapter for the real sibling ml.inference.predictor.SignPredictor."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.predictor: Any = None
        self.error: str | None = "Model has not been loaded"
        self._unavailable_exceptions: tuple[type[BaseException], ...] = (ModelUnavailableError,)
        self._prediction_cache: OrderedDict[str, tuple[str, float, dict[str, Any]]] = OrderedDict()

    @property
    def available(self) -> bool:
        return self.predictor is not None

    def reload(self) -> None:
        """Re-scan the models directory; used after training new artifacts."""
        self.predictor = None
        self.load()

    def load(self) -> None:
        if str(BACKEND_DIR) not in sys.path:
            sys.path.insert(0, str(BACKEND_DIR))
        try:
            module = importlib.import_module("ml.inference.predictor")
            external_error = getattr(module, "ModelUnavailableError", ModelUnavailableError)
            self._unavailable_exceptions = (ModelUnavailableError, external_error)
            predictor_class = module.SignPredictor
            signature = inspect.signature(predictor_class)
            if "models_dir" in signature.parameters:
                self.predictor = predictor_class(models_dir=self.settings.models_dir)
            elif "model_dir" in signature.parameters:
                self.predictor = predictor_class(model_dir=self.settings.models_dir)
            else:
                self.predictor = predictor_class()
            self.error = None
        except Exception as exc:
            self.predictor = None
            self.error = str(exc) or exc.__class__.__name__
            logger.warning("Sign recognition model is unavailable: %s", self.error)

    async def predict(
        self,
        image_bytes: bytes,
        *,
        landmarks: Any = None,
        sequence: Any = None,
    ) -> dict[str, Any]:
        if self.predictor is None:
            raise ModelUnavailableError(self.error or "Sign recognition model is unavailable")
        method = self.predictor.predict
        kwargs = {
            key: value
            for key, value in {"landmarks": landmarks, "sequence": sequence}.items()
            if value is not None
        }
        if kwargs:
            parameters = inspect.signature(method).parameters
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            unsupported = set(kwargs) - set(parameters)
            if unsupported and not accepts_kwargs:
                names = ", ".join(sorted(unsupported))
                raise ValueError(f"SignPredictor does not support: {names}")
        try:
            if inspect.iscoroutinefunction(method):
                result = await method(image_bytes, **kwargs)
            else:
                result = await asyncio.to_thread(method, image_bytes, **kwargs)
        except self._unavailable_exceptions as exc:
            raise ModelUnavailableError(str(exc) or "Sign recognition model is unavailable") from exc
        return self._normalize_prediction(result)

    async def predict_sequences(self, sequences: list[Any]) -> list[dict[str, Any]]:
        """Classify many landmark windows in one batched model call."""
        if self.predictor is None:
            raise ModelUnavailableError(self.error or "Sign recognition model is unavailable")
        method = getattr(self.predictor, "predict_sequences", None)
        if method is None:
            raise ModelUnavailableError("The loaded model does not support batch sequence prediction")
        try:
            results = await asyncio.to_thread(method, sequences)
        except self._unavailable_exceptions as exc:
            raise ModelUnavailableError(str(exc) or "Sign recognition model is unavailable") from exc
        if not isinstance(results, (list, tuple)):
            raise ValueError("SignPredictor returned an invalid batch result")
        return [self._normalize_prediction(result) for result in results]

    @staticmethod
    def _normalize_prediction(result: Any) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise ValueError("SignPredictor returned a non-object result")

        raw = dict(result)
        sign = raw.get("sign") or raw.get("label") or raw.get("prediction")
        if not isinstance(sign, str):
            raise ValueError("SignPredictor result is missing a sign")
        normalized = {
            "sign": sign,
            "confidence": raw.get("confidence", raw.get("score", raw.get("probability"))),
            "category": raw.get("category"),
            "model_version": raw.get("model_version", raw.get("modelVersion")),
            "recognition_type": raw.get("recognition_type"),
            "top_k": raw.get("top_k", raw.get("top3", raw.get("alternatives", []))),
            "hand_detected": raw.get("hand_detected"),
            "landmarks": raw.get("landmarks", raw.get("hand_landmarks")),
        }
        try:
            return PredictionResult.model_validate(normalized).model_dump()
        except ValidationError as exc:
            raise ValueError("SignPredictor returned an invalid prediction") from exc

    def cache_prediction(self, user_id: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        now = time.monotonic()
        self._purge_predictions(now)
        prediction_id = secrets.token_urlsafe(24)
        cached = dict(prediction)
        cached.pop("prediction_id", None)
        self._prediction_cache[prediction_id] = (
            str(user_id),
            now + self.settings.prediction_ttl_seconds,
            cached,
        )
        while len(self._prediction_cache) > self.settings.prediction_cache_size:
            self._prediction_cache.popitem(last=False)
        return {**cached, "prediction_id": prediction_id}

    def cached_prediction(self, prediction_id: str, user_id: Any) -> dict[str, Any] | None:
        entry = self._prediction_cache.get(prediction_id)
        if entry is None:
            return None
        cached_user_id, expires_at, prediction = entry
        if expires_at <= time.monotonic() or cached_user_id != str(user_id):
            if expires_at <= time.monotonic():
                self._prediction_cache.pop(prediction_id, None)
            return None
        self._prediction_cache.move_to_end(prediction_id)
        return dict(prediction)

    def _purge_predictions(self, now: float) -> None:
        expired = [
            prediction_id
            for prediction_id, (_, expires_at, _) in self._prediction_cache.items()
            if expires_at <= now
        ]
        for prediction_id in expired:
            self._prediction_cache.pop(prediction_id, None)
