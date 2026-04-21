import os
import sys
from yaml import safe_load
import random
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from typing import Dict
import argparse
import pynvml


# Ensure deterministic behaviour
def set_seed(seed, logger):
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


# Ensure configs are merged recursively
def deep_merge(a: dict, b: dict) -> dict:
    result = a.copy()
    for key, value in b.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_num_training_steps(dataloader: DataLoader, config_dict: Dict):
    return config_dict['training']['num_epochs'] * (len(dataloader) // config_dict['training']['grad_accumulation_steps'])


def get_num_warmup_steps(num_training_steps: int) -> int:
    return int(0.03 * num_training_steps)


def parse_args():
    parser = argparse.ArgumentParser(description="Project: peft_lora_eval")

    parser.add_argument("--run_id", required=True)
    parser.add_argument("--config", default="configs/run.yaml")

    parser.add_argument("--methods", type=str, default="base", help="Method name (ex: 'base/LoRA/QLoRA/QDoRA')")
    parser.add_argument("--stages", type=str, default="eval", help="Str in this frmt: train/eval/inf/bench")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")

    parser.add_argument("--base-path", type=str, default="./", help="Current dir")
    parser.add_argument("--data-subset", type=int, default=None, help="Subset of the dataset to use")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR/CRITICAL")

    parser.add_argument("--merge", action="store_true", help="Merge adapter into the base model for eval/inference")
    parser.add_argument("--use-small-model", action="store_true", help="Use gpt-2")
    return parser.parse_args()


def verify_parsed_args(args):
    ### NOT COMPLETE ###
    # Verify arg.methods
    allowed_methods = ['lora', 'qlora', 'dora', 'qdora', 'base']
    methods = args.methods.lower().strip().split("/")

    for method in methods:
        if method not in allowed_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from: {'/'.join(allowed_methods)}")

    # Verify arg.stages
    allowed_stages = ['eval', 'inf', 'train', 'bench']
    stages = args.stages.lower().strip().split("/")

    for stage in stages:
        if stage not in allowed_stages:
            raise ValueError(f"Invalid stage '{stage}'. Choose from: {'/'.join(allowed_stages)}")

    if args.seed is not None and not isinstance(args.seed, int):
        raise ValueError(f"Invalid seed '{args.seed}'. Must be integer.")

    return args


def load_and_validate_run_config(args, logger):
    base_path = args.base_path
    config_path = args.config
    logger.debug(f"Run config path: {config_path}.")

    args = verify_parsed_args(args)

    with open(os.path.join(base_path, config_path), "r") as f:
        config = safe_load(f)
        config["methods"] = args.methods.lower().strip().split("/")
        config["stages"] = args.stages.lower().strip().split("/")

        if args.wandb_project is not None:
            config["logging"]["wandb_project"] = args.wandb_project
        if args.data_subset is not None:
            config["dataset"]["subset"] = args.data_subset
        if args.seed is not None:
            config["runtime"]["seed"] = args.seed

        config["runtime"]["base_path"] = args.base_path
        config["runtime"]["merge"] = args.merge
        config["runtime"]["use_small_model"] = args.use_small_model
    return config


def load_method_config(method, run_config, logger):
    base_path = run_config["runtime"]["base_path"]
    config_path = os.path.join(base_path, f"configs/methods/{method}.yaml")
    logger.debug(f"Method config path: {config_path}.")

    with open(config_path, "r") as f:
        method_config = safe_load(f)

    config = deep_merge(run_config, method_config)
    config["active_method"] = method

    return config


def in_colab() -> bool:
    """Checks if the current environment is Google Colab."""
    return "google.colab" in sys.modules


def in_kaggle() -> bool:
    """Checks if the current environment is Kaggle Notebooks."""
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def select_attn_implementation() -> str | None:
    """
    Auto-select the best attention implementation.
    Safe for Colab + new pynvml (returns str, not bytes).
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name_bytes_or_str = pynvml.nvmlDeviceGetName(handle)

        # Handle both old (bytes) and new (str) return types
        if isinstance(name_bytes_or_str, bytes):
            gpu_name = name_bytes_or_str.decode("utf-8")
        else:
            gpu_name = str(name_bytes_or_str)

        gpu_name = gpu_name.lower()

        if "a100" in gpu_name or "a10" in gpu_name or "h100" in gpu_name:
            return "flash_attention_2"
        elif "rtx 4090" in gpu_name or "rtx 4080" in gpu_name or "l4" in gpu_name:
            return "flash_attention_2"
        else:
            return "eager"  # safe fallback

    except Exception as e:
        print(f"Failed to detect GPU for attention: {e}. Using eager.")
        return "eager"


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_base_path(args):
    if in_colab():
        p = "/content/drive/MyDrive/peft_lora_eval/"
        return p
    elif in_kaggle():
        p = "/kaggle/input/"
        return p
    else:
        return args.base_path