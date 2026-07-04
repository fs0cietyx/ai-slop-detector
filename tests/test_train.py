from unittest.mock import MagicMock, patch

import numpy as np

from slopguard.cli.train_model import compute_metrics, run_training


def test_compute_metrics() -> None:
    """Test the metric computation logic."""
    # Dummy eval_pred tuple: (logits, labels)
    # 2 samples, 2 classes
    logits = np.array(
        [
            [2.0, -1.0],  # Prediction: 0
            [-1.0, 2.0],  # Prediction: 1
        ]
    )
    labels = np.array([0, 1])

    # We don't want to actually load HF evaluate, so we patch it
    with patch("slopguard.cli.train_model.evaluate.load") as mock_load:
        mock_metric = MagicMock()
        mock_metric.compute.return_value = {"accuracy": 1.0, "f1": 1.0}
        mock_load.return_value = mock_metric

        result = compute_metrics((logits, labels))

        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
        assert mock_load.call_count == 2  # accuracy and f1


def test_run_training_no_data() -> None:
    """Test training aborts if no data is found."""
    with patch("os.path.exists", return_value=False):
        # Should return silently (with an error log)
        run_training()


@patch("os.path.exists", return_value=True)
@patch("slopguard.cli.train_model.pd.read_csv")
@patch("slopguard.cli.train_model.Dataset.from_pandas")
@patch("slopguard.cli.train_model.AutoTokenizer.from_pretrained")
@patch("slopguard.cli.train_model.AutoModelForSequenceClassification.from_pretrained")
@patch("slopguard.cli.train_model.get_peft_model")
@patch("slopguard.cli.train_model.Trainer")
def test_run_training_success(
    mock_trainer: MagicMock,
    mock_get_peft: MagicMock,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
    mock_dataset: MagicMock,
    mock_read_csv: MagicMock,
    mock_exists: MagicMock,
) -> None:
    """Test successful training initialization and execution."""
    # Setup dataframe mock
    mock_df = MagicMock()
    mock_df.__getitem__.return_value.astype.return_value = ["dummy"]
    # We must mock the boolean masking operation df[df["label"].isin(...)]
    mock_df.__getitem__.return_value.isin.return_value = [True]
    # Reassigning mock_df to simulate the df = df[...] return
    mock_read_csv.return_value = mock_df

    # Setup Dataset mock
    mock_hf_dataset = MagicMock()
    mock_dataset.return_value = mock_hf_dataset
    mock_hf_dataset.rename_column.return_value = mock_hf_dataset
    mock_hf_dataset.train_test_split.return_value = mock_hf_dataset
    mock_hf_dataset.map.return_value = {"train": [], "test": []}

    # Execute
    run_training()

    # Assert
    mock_read_csv.assert_called_once_with("data/training_dataset.csv")
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch("slopguard.cli.train_model.pd.read_csv", side_effect=Exception("CSV Error"))
def test_run_training_csv_error(mock_read_csv: MagicMock, mock_exists: MagicMock) -> None:
    """Test training aborts on CSV read error."""
    run_training()
    # Should safely return
