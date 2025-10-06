import json
import pytest
from unittest.mock import mock_open, patch
from datetime import datetime as RealDatetime

from src.evaluator import compute_metrics, save_results


@pytest.fixture
def mock_metrics_nonempty():
    """Mock metrics dict with non-empty lists."""
    return {
        "batches": [0, 1, 2],
        "losses": [0.5, 1.0, 0.8],
        "perplexities": [1.648721, 2.718282, 2.22554],
        "vram": [1000.0, 1500.0, 1200.0],
    }


@pytest.fixture
def mock_metrics_empty():
    """Mock metrics dict with empty lists."""
    return {
        "batches": [],
        "losses": [],
        "perplexities": [],
        "vram": [],
    }


def test_compute_metrics_nonempty(mock_metrics_nonempty):
    """Test compute_metrics with non-empty data."""
    result = compute_metrics(mock_metrics_nonempty)

    assert result["batch"] == mock_metrics_nonempty["batches"]
    assert len(result["evaluation"]["losses"]) == len(mock_metrics_nonempty["losses"])

    # Check computed values
    assert result["evaluation"]["max_loss"] == 1.0
    assert result["evaluation"]["min_loss"] == 0.5
    assert result["evaluation"]["avg_loss"] == pytest.approx((0.5 + 1.0 + 0.8) / 3, abs=1e-6)
    assert result["evaluation"]["perplexity"] == pytest.approx(
        sum(mock_metrics_nonempty["perplexities"]) / 3, abs=1e-6
    )

    assert result["hardware"]["avg_vram"] == pytest.approx(
        sum(mock_metrics_nonempty["vram"]) / 3, abs=1e-6
    )
    assert result["hardware"]["peak_vram"] == 1500.0


def test_compute_metrics_empty(mock_metrics_empty):
    """Test compute_metrics with empty lists (edge case)."""
    result = compute_metrics(mock_metrics_empty)

    assert result["batch"] == []
    assert result["evaluation"]["max_loss"] == 0
    assert result["evaluation"]["min_loss"] == 0
    assert result["evaluation"]["avg_loss"] == 0
    assert result["evaluation"]["perplexity"] == 0
    assert result["hardware"]["avg_vram"] == 0
    assert result["hardware"]["peak_vram"] == 0


@pytest.fixture
def mock_results_data():
    """Mock data for save_results test."""
    return {
        "metadata": {"eval_date": "2025-10-05", "dataset_split": "val"},
        "config": {"model": {"model_name_or_path": "test-model"}},
        "metrics": {"evaluation": {"avg_loss": 0.7}},
    }


@patch("builtins.open", new_callable=mock_open)
@patch("src.evaluator.logger")
def test_save_results(mock_logger, mock_file, mock_results_data):
    """Test save_results with mocked file I/O."""
    mock_test_datetime = RealDatetime(2025, 10, 5, 12, 0, 0)
    expected_timestamp_strftime = mock_test_datetime.strftime("%Y%m%d_%H%M%S")
    expected_filename = f"results/metrics/test-model_{expected_timestamp_strftime}.json"
    mock_results_data["timestamp"] = mock_test_datetime.isoformat()

    with patch("src.evaluator.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_test_datetime
        save_results(
            mock_results_data["metrics"],
            mock_results_data["metadata"],
            mock_results_data["config"],
        )

    # Check file open call
    mock_file.assert_called_once_with(expected_filename, "w")
    handle = mock_file.return_value
    # Since json.dump may call write multiple times due to indentation, check that it's called
    assert handle.write.called
    # For a simple check, verify the content was written (json.dump handles the writes)
    expected_content = json.dumps(mock_results_data, indent=2)
    # json.dump writes in chunks, so we can check if any write call contains the expected json
    write_calls = [call.args[0] for call in handle.write.call_args_list]
    full_content = "".join(write_calls)
    assert expected_content in full_content  # Rough check; for exact, could patch json.dump

    mock_logger.info.assert_called_once_with(f"Results saved to {expected_filename}")