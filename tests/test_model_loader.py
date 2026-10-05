from unittest.mock import patch

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