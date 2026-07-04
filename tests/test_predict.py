import sys
from unittest.mock import patch

from slopguard.cli.predict import run_cli


def test_predict_cli_no_args() -> None:
    """Test the raw CLI script with no arguments."""
    with patch.object(sys, "argv", ["predict.py"]):
        with patch("sys.exit", side_effect=SystemExit) as mock_exit:
            try:
                run_cli()
            except SystemExit:
                pass
            mock_exit.assert_called_once_with(0)


def test_predict_cli_with_args() -> None:
    """Test the raw CLI script with arguments and mocked engine."""
    with patch.object(sys, "argv", ["predict.py", "Test payload"]):
        with patch("slopguard.cli.predict.InferenceEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine._device.type = "cpu"
            mock_engine.predict.return_value = ("AI-GENERATED", 0.98)

            # Prevent it from calling sys.exit or throwing an exception
            run_cli()
            mock_engine.predict.assert_called_once_with("Test payload")


def test_predict_cli_keyboard_interrupt() -> None:
    """Test graceful shutdown on keyboard interrupt."""
    with patch.object(sys, "argv", ["predict.py", "Test payload"]):
        with patch("slopguard.cli.predict.InferenceEngine", side_effect=KeyboardInterrupt()):
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                try:
                    run_cli()
                except SystemExit:
                    pass
                mock_exit.assert_called_once_with(0)


def test_predict_cli_exception() -> None:
    """Test graceful handling of fatal errors."""
    with patch.object(sys, "argv", ["predict.py", "Test payload"]):
        with patch("slopguard.cli.predict.InferenceEngine", side_effect=Exception("Disk full")):
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                try:
                    run_cli()
                except SystemExit:
                    pass
                mock_exit.assert_called_once_with(1)
