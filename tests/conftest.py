from unittest.mock import MagicMock, patch

# Mock load_assets globally during test collection
patch("slopguard.core.engine.InferenceEngine._load_assets", return_value=(MagicMock(), MagicMock())).start()
