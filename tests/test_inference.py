import pytest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from src.inference import compute_metrics, decode_generated_response, run_inference


class ExitException(Exception):
    pass


class _BatchEncoding(dict):
    def __init__(self, input_ids):
        super().__init__({"input_ids": input_ids})
        self.input_ids = input_ids

    def to(self, device):
        return self


def test_compute_metrics():
    temp_metrics = {
        "latencies_ms": [100, 200, 300],
        "input_time": 0.5,
        "input_token_count": 10,
        "peak_vram": 3.0,
        "vram_usage": [1.0, 2.0, 3.0],
        "ram_total": 16.0,
        "ram_usage": [4.0, 5.0, 6.0],
        "total_input_tokens": 15,
        "total_output_tokens": 20,
        "energy_consumption": 0.1,
        "wall_clock_time": 10.0,
    }
    metrics = compute_metrics(temp_metrics)
    assert metrics["inference_latency"]["p50_latency"] == 200
    assert metrics["inference_latency"]["p95_latency"] == 290.0
    assert metrics["memory_usage"]["peak_vram"] == 3.0
    assert metrics["memory_usage"]["avg_vram"] == 2.0
    assert "time_and_compute" in metrics


def test_decode_generated_response_excludes_prompt_tokens():
    tokenizer = MagicMock()
    generated_ids = torch.tensor([[10, 11, 12, 13, 14]])
    tokenizer.batch_decode.return_value = ["new text"]

    response = decode_generated_response(generated_ids, 3, tokenizer)

    assert response == "new text"
    tokenizer.batch_decode.assert_called_once()
    decoded_tokens = tokenizer.batch_decode.call_args.args[0]
    assert torch.equal(decoded_tokens, torch.tensor([[13, 14]]))


@pytest.fixture
def mock_dependencies():
    with (
        patch("builtins.input", side_effect=["Hello", "STOP"]),
        patch("sys.exit") as mock_exit,
        patch("src.inference.wandb.log"),
        patch("src.inference.logger.info"),
        patch("src.inference.logger.debug"),
        patch("psutil.virtual_memory") as mock_mem,
        patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_handle,
        patch("pynvml.nvmlDeviceGetPowerUsage", side_effect=[100000, 100000, 100000]),
        patch("pynvml.nvmlShutdown"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.cuda.max_memory_allocated", return_value=4e9),
        patch("torch.is_tensor", return_value=True),
    ):
        mock_mem.return_value = MagicMock(total=16e9, used=8e9)
        mock_handle.return_value = MagicMock()
        yield mock_exit


def test_run_inference(mock_dependencies):
    mock_exit = mock_dependencies
    mock_exit.side_effect = ExitException

    mock_model = MagicMock()
    mock_model.device = "cuda:0"
    mock_param = MagicMock(device="cuda:0")
    mock_model.parameters.return_value = iter([mock_param])
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_input_ids = torch.tensor([[1, 2, 3]])
    mock_batch_encoding = _BatchEncoding(mock_input_ids)
    mock_tokenizer.return_value = mock_batch_encoding
    mock_tokenizer.batch_decode.return_value = ["Response"]

    with pytest.raises(ExitException):
        with (
            patch("time.time", side_effect=[0, 10]),
            patch(
                "time.perf_counter",
                side_effect=[0.0, 1.0, 1.01, 1.02, 1.1, 1.2, 1.3, 1.4, 1.41],
            ),
        ):
            run_inference(mock_model, mock_tokenizer, metadata={})

    mock_model.generate.assert_called_once_with(
        **mock_batch_encoding,
        num_beams=4,
        do_sample=True,
        pad_token_id=0,
    )
    mock_tokenizer.batch_decode.assert_called_once()
    mock_exit.assert_called_once_with(0)
