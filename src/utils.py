import os
from yaml import safe_load
import random
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from typing import Dict

from scripts.run_train import verify_parsed_args
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


def load_and_validate_config(args):
    root_dir = "./"
    config_path = os.path.join("configs/", f"{args.method.lower()}_config.yaml")
    logger.info(f"config_path: {config_path}")

    # verify arguments
    verify_parsed_args(args)

    with open(os.path.join(root_dir, config_path), "r") as f:
        config_dict = safe_load(f)
        config_dict['mode'] = args.mode.lower()
        config_dict['method'] = args.method.lower()
        config_dict['seed'] = args.seed
        config_dict['output_path'] = args.output_path
        config_dict['data_subset'] = args.data_subset

    logger.info(f"Parsed args: \nconfig_path: {config_path}\nmethod: {args.method}\nseed: {args.seed}\n\n"
                 f"Config_dict: \n{config_dict}")
    return config_dict