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


@patch("ml.lifecycle.MlflowClient")
def test_model_lifecycle_register_promote_and_rollback(
    mock_client_class,
) -> None:
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate registration returning version 2.
    registered_version = "2"

    # Promote version 2 to production.
    mock_client.set_registered_model_alias(
        "ticket_classifier",
        "production",
        registered_version,
    )

    # Roll back production to version 1.
    rollback_model(
        model_name="ticket_classifier",
        target_version="1",
        alias="production",
    )

    mock_client.set_registered_model_alias.assert_any_call(
        "ticket_classifier",
        "production",
        "2",
    )
    mock_client.set_registered_model_alias.assert_any_call(
        "ticket_classifier",
        "production",
        "1",
    )
