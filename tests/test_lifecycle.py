from unittest.mock import MagicMock, patch

from ml.lifecycle import rollback_model


@patch("ml.lifecycle.MlflowClient")
def test_rollback_moves_production_alias(mock_client_class) -> None:
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    rollback_model(
        model_name="ticket_classifier",
        target_version="1",
        alias="production",
    )

    mock_client.set_registered_model_alias.assert_called_once_with(
        "ticket_classifier",
        "production",
        "1",
    )