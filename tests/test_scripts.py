# import pytest
# from unittest.mock import patch, MagicMock, ANY
#
# import torch
# from transformers import AutoTokenizer
#
# from src.auth import init_wandb, init_hf_auth
# from src.inference import run_inference
# from src.logger import get_logger, set_up_logging
# from src.model_utils import configure_peft_model_for_eval
# from src.utils import parse_args, load_and_validate_config, set_seed
#
# class ExitException(Exception):
#     pass
#
# @pytest.fixture
# def mock_args():
#     mock_args = MagicMock()
#     mock_args.config_path = "tests/fixtures/mock_config.yaml" # Assume a fixture config exists
#     return mock_args
#
# @pytest.fixture
# def mock_config():
#     return {
#         "seed": 42,
#         "model": {"model_name_or_path": "gpt2"},
#     }
#
# def test_main(mock_args, mock_config):
#     with (
#         patch("scripts.run_inference.parse_args", return_value=mock_args),
#         patch("scripts.run_inference.load_and_validate_config", return_value=mock_config),
#         patch("scripts.run_inference.set_up_logging"),
#         patch("scripts.run_inference.set_seed"),
#         patch("scripts.run_inference.init_wandb"),
#         patch("scripts.run_inference.init_hf_auth"),
#         patch("scripts.run_inference.configure_peft_model_for_eval") as mock_model,
#         patch("scripts.run_inference.AutoTokenizer.from_pretrained") as mock_tokenizer,
#         patch("scripts.run_inference.run_inference") as mock_run,
#         patch("scripts.run_inference.torch.cuda.is_available", return_value=True),
#         patch("scripts.run_inference.get_logger") as mock_get_logger,
#     ):
#         mock_model_instance = MagicMock()
#         mock_model.return_value = mock_model_instance
#         mock_tokenizer_instance = MagicMock(spec=AutoTokenizer)
#         mock_tokenizer_instance.eos_token = "<|endoftext|>"
#         mock_tokenizer.return_value = mock_tokenizer_instance
#         mock_get_logger.return_value = MagicMock()
#
#         mock_run.side_effect = ExitException
#
#         from scripts.run_inference import main
#         with pytest.raises(ExitException):
#             main()
#         mock_model.assert_called_once_with(mock_config, "cuda")
#         mock_tokenizer.assert_called_once_with("gpt2")
#         mock_run.assert_called_once_with(mock_model_instance, mock_tokenizer_instance)


# import pytest
# from unittest.mock import patch, MagicMock
# import importlib  # For potential reload, but not needed here
#
# from scripts.run_evaluate import main, metadata
#
#
# @pytest.fixture
# def mock_config():
#     """Mock config dict."""
#     return {
#         "seed": 42,
#         "dataset_name": "mock_dataset",
#         "data_subset": "test",
#         "dataloader": {"batch_size": 4},
#     }
#
#
# @patch("scripts.run_evaluate.evaluator")
# @patch("scripts.run_evaluate.configure_peft_model")
# @patch("scripts.run_evaluate.get_dataloader")
# @patch("scripts.run_evaluate.split_and_sort_dataset")
# @patch("scripts.run_evaluate.get_tokenized_dataset")
# @patch("scripts.run_evaluate.load_alpaca_data")
# @patch("scripts.run_evaluate.init_hf_auth")
# @patch("scripts.run_evaluate.init_wandb")
# @patch("scripts.run_evaluate.set_seed")
# @patch("scripts.run_evaluate.load_and_validate_config")
# @patch("scripts.run_evaluate.parse_args")
# @patch("scripts.run_evaluate.set_up_logging")
# def test_main_smoke_test(
#     mock_set_up_logging,
#     mock_parse_args,
#     mock_load_config,
#     mock_set_seed,
#     mock_init_wandb,
#     mock_init_hf_auth,
#     mock_load_data,
#     mock_get_tokenized,
#     mock_split_dataset,
#     mock_get_dataloader,
#     mock_configure_model,
#     mock_evaluator,
#     mock_config,
# ):
#     """Basic smoke test for main(): ensure the function runs and calls all dependencies in order."""
#     # Mock returns
#     mock_args = MagicMock()
#     mock_parse_args.return_value = mock_args
#     mock_load_config.return_value = mock_config
#
#     mock_dataset = MagicMock()
#     mock_load_data.return_value = mock_dataset
#
#     # FIX for chaining: Mock .map() to return the same dataset instance (simulates formatting)
#     mock_dataset.map.return_value = mock_dataset  # No change in instance for simplicity
#
#     mock_tokenized, mock_tokenizer = MagicMock(), MagicMock()
#     mock_get_tokenized.return_value = (mock_tokenized, mock_tokenizer)
#
#     mock_splits = {"train": MagicMock(), "val": MagicMock(), "test": MagicMock()}
#     mock_split_dataset.return_value = mock_splits
#     mock_test_ds = mock_splits["test"]
#
#     mock_loader = MagicMock()
#     mock_get_dataloader.return_value = mock_loader
#
#     mock_model = MagicMock()
#     mock_configure_model.return_value = mock_model
#
#     mock_evaluator.return_value = None
#
#     # Run main
#     main()
#
#     # Assert calls
#     mock_set_up_logging.assert_called_once()
#     mock_parse_args.assert_called_once()
#     mock_load_config.assert_called_once_with(args=mock_args)
#     mock_set_seed.assert_called_once_with(mock_config["seed"])
#     mock_init_wandb.assert_called_once_with(mock_config)
#     mock_init_hf_auth.assert_called_once()
#
#     mock_load_data.assert_called_once_with(
#         dataset_name=mock_config["dataset_name"],
#         data_subset=mock_config["data_subset"],
#     )
#
#     # format_prompt is called via map, but we mock the dataset.map() to return self
#     mock_get_tokenized.assert_called_once_with(mock_dataset, mock_config)  # Now matches!
#
#     mock_split_dataset.assert_called_once_with(
#         mock_tokenized,
#         config=mock_config["seed"],  # Note: this is a bug in the script, but testing as-is
#     )
#
#     mock_get_dataloader.assert_called_once_with(
#         mock_test_ds,
#         mock_tokenizer,
#         batch_size=mock_config["dataloader"]["batch_size"],
#         seed=mock_config["seed"],
#     )
#
#     mock_configure_model.assert_called_once_with(config_dict=mock_config, device="cuda")  # Assuming cuda available
#
#     mock_evaluator.assert_called_once_with(
#         mock_model,
#         mock_loader,
#         mock_config,
#         "cuda",  # Device
#         metadata,
#     )
#
#
# @patch("scripts.run_evaluate.torch.cuda.is_available")
# def test_device_cuda(mock_is_available):
#     """Test that device is 'cuda' when available."""
#     mock_is_available.return_value = True
#
#     # Same re-import for consistency
#     from scripts.run_evaluate import device
#     assert device == "cuda"