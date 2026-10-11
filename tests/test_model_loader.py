from unittest.mock import patch

import pytest

from app.model_loader import ModelLoader


def test_is_healthy_returns_true_when_model_load_succeeds():
    loader = ModelLoader()
    with patch.object(loader, "load", return_value=object()):
        assert loader.is_healthy() is True
def test_is_healthy_returns_false_when_model_load_fails():
    loader = ModelLoader()
    with patch.object(
        loader,
        "load",
        side_effect=Exception("Model unavailable"),
    ):
        assert loader.is_healthy() is False

def test_empty_tracking_uri_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        "app.model_loader.MLFLOW_TRACKING_URI",
        " ",
    )
    with pytest.raises(
        ValueError,
        match="MLFLOW_TRACKING_URI must not be empty",
    ):
        ModelLoader()


def test_empty_model_name_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        "app.model_loader.MODEL_NAME",
        "",
    )
    with pytest.raises(
        ValueError,
        match="MLFLOW_MODEL_NAME must not be empty",
    ):
        ModelLoader()


def test_empty_model_alias_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        "app.model_loader.MODEL_ALIAS",
        "",
    )
    with pytest.raises(
        ValueError,
        match="MLFLOW_MODEL_ALIAS must not be empty",
    ):
        ModelLoader()