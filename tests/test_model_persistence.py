import torch

from ml.model import build_model


def test_model_can_be_saved_and_loaded(tmp_path) -> None:
    model = build_model()
    model.eval()

    inputs = torch.randn(8, 4)

    with torch.no_grad():
        original_outputs = model(inputs)

    model_path = tmp_path / "model.pt"

    torch.save(model.state_dict(), model_path)

    loaded_model = build_model()
    loaded_model.load_state_dict(
        torch.load(model_path, weights_only=True)
    )
    loaded_model.eval()

    with torch.no_grad():
        loaded_outputs = loaded_model(inputs)

    assert loaded_outputs.shape == (8, 1)
    assert torch.all(torch.isfinite(loaded_outputs))
    assert torch.all(loaded_outputs >= 0.0)
    assert torch.all(loaded_outputs <= 1.0)

    torch.testing.assert_close(
        original_outputs,
        loaded_outputs,
    )