import random
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from typing import Dict
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
    return config_dict['num_epochs'] * (len(dataloader) // config_dict['grad_accumulation_steps'])


def get_num_warmup_steps(num_training_steps: int) -> int:
    return int(0.03 * num_training_steps)


