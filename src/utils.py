import os
import sys

from sympy.logic.boolalg import Boolean
from yaml import safe_load
import random
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from typing import Dict
import argparse

from src.logger import get_logger

logger = get_logger()


# Ensure deterministic behaviour
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    # Ensure deterministic behaviour on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Warn if non-deterministic algorithms are used
    torch.use_deterministic_algorithms(True, warn_only=True)  # warning won't be logged

    logger.info(f"Seed is set to '{seed}'")


def get_num_training_steps(dataloader: DataLoader, config_dict: Dict):
    return config_dict['training']['num_epochs'] * (len(dataloader) // config_dict['training']['grad_accumulation_steps'])


def get_num_warmup_steps(num_training_steps: int) -> int:
    return int(0.03 * num_training_steps)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 3B fine-tune")

    # add arguments
    parser.add_argument("--use-small-model", help="True to use smaller model (gpt-2) for debugging.")
    parser.add_argument("--mode", type=str, default="test", help="Mode: train/test (train for training, test for testing the pipeline")
    parser.add_argument("--method", type=str, default="lora", help="Method name (LoRA/QLoRA/QDoRA)")
    parser.add_argument("--merge", type=bool, default="False", help="Merge base model with adapter.")
    parser.add_argument("--seed", type=int, default=17, help="Seed number")
    parser.add_argument("--base-path", type=str, default="./", help="Current dir")
    parser.add_argument("--output-path", type=str, default=r"models\<method>_best", help="Checkpoint output path")
    parser.add_argument("--wandb-project", type=str, default="llama-finetune",
                        help="Specify the WandB project name for experiment tracking")
    parser.add_argument("--data-subset", type=int, default=100, help="Subset of the dataset to use")

    logger.info("Successfully parsed args")

    return parser.parse_args()


def verify_parsed_args(args):
    # NOT COMPLETE
    # verify argument parsing
    if args.method.lower() not in ['lora', 'qlora', 'dora', 'qdora', 'base']:
        msg = f"Invalid method '{args.method}'. Choose from: lora, qlora, qdora, base."
        logger.error(msg)
        raise ValueError(f"Invalid method '{args.method}'. Choose from: lora/qlora/qdora/base.")

    if not isinstance(args.seed, int):
        raise ValueError(f"Invalid seed '{args.seed}'. Must be integer.")


def load_and_validate_config(args):
    base_path = args.base_path
    config_path = os.path.join("configs/", f"{args.method.lower()}_config.yaml")
    logger.info(f"config_path: {config_path}")

    # verify arguments
    verify_parsed_args(args)

    with open(os.path.join(base_path, config_path), "r") as f:
        config_dict = safe_load(f)
        config_dict["use_small_model"] = args.use_small_model
        config_dict['mode'] = args.mode.lower()
        config_dict['method'] = args.method.lower()
        config_dict["merge"] = args.merge
        config_dict['seed'] = args.seed
        config_dict['output_path'] = args.output_path
        config_dict['data_subset'] = args.data_subset
        config_dict['base_path'] = base_path

    logger.info(f"Parsed args: \nconfig_path: {config_path}\nmethod: {args.method}\nseed: {args.seed}\n\n"
                 f"Config_dict: \n{config_dict}")
    return config_dict


def check_if_checkpoints_exist(config: Dict):
    # use only in run_evaluate.py
    if os.path.exists(config["output_path"]):
        logger.debug(f"Checkpoints exist at '{config["output_path"]}'")
    else:
        config["method"] = "base"
        logger.warning(f"No checkpoints at '{config["output_path"]}'.\n"
                       f"Only 'base' method is available. config['method'] set to {config['method']}.")
        if (input("Do you want to proceed with method set to 'base'? Y/n?")).strip().lower() not in ['y', 'yes']:
            sys.exit(1)


def in_colab() -> bool:
    """Checks if the current environment is Google Colab."""
    return "google.colab" in sys.modules