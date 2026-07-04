from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from slopguard.cli.main import app

runner = CliRunner()


def test_cli_predict() -> None:
    """Test the predict CLI command with mocked engine."""
    # We need to mock the InferenceEngine in its actual module since it's a delayed import
    with patch("slopguard.core.engine.InferenceEngine") as MockEngine:
        mock_instance = MockEngine.return_value
        mock_instance.predict.return_value = ("HUMAN-WRITTEN", 0.99)
        
        result = runner.invoke(app, ["predict", "This is human written text."])
        
        assert result.exit_code == 0
        # The output might not capture logger.info depending on Typer/Logging setup,
        # but we can at least assert no exceptions were thrown and exit code is 0.
        mock_instance.predict.assert_called_once_with("This is human written text.")


def test_cli_train() -> None:
    """Test the train CLI command."""
    with patch("slopguard.cli.train_model.run_training") as mock_run_training:
        result = runner.invoke(app, ["train"])
        assert result.exit_code == 0
        mock_run_training.assert_called_once()
