from __future__ import annotations

import pytest

from ml.inference.predictor import ModelUnavailableError, SignPredictor


def test_missing_models_directory_raises(tmp_path):
    with pytest.raises(ModelUnavailableError):
        SignPredictor(models_dir=tmp_path / "does-not-exist")


def test_empty_models_directory_raises(tmp_path):
    with pytest.raises(ModelUnavailableError, match="No trained model"):
        SignPredictor(models_dir=tmp_path)
