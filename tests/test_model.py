import torch

from ml.model import build_model


def test_model_accepts_four_features():
    model = build_model()

    inputs = torch.randn(8, 4)
    outputs = model(inputs)

    assert outputs.shape == (8, 1)


def test_model_outputs_probabilities():
    model = build_model()

    inputs = torch.randn(8, 4)
    outputs = model(inputs)

    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)


def test_model_produces_finite_outputs():
    model = build_model()

    inputs = torch.randn(8, 4)
    outputs = model(inputs)

    assert torch.isfinite(outputs).all()