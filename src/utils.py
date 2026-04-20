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


def get_num_training_steps(dataloader: DataLoader, config_dict: Dict):
    return config_dict['training']['num_epochs'] * (len(dataloader) // config_dict['training']['grad_accumulation_steps'])


def get_num_warmup_steps(num_training_steps: int) -> int:
    return int(0.03 * num_training_steps)


def parse_args():
    parser = argparse.ArgumentParser(description="Project: peft_lora_eval")

    parser.add_argument("--run_id", help="Run identification number")
    parser.add_argument("--log-level", default="INFO", help="Level of logging (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
    parser.add_argument("--stages", type=str, default="eval", help="String of all stages in this format:"
                        "train/eval/inf/bench")
    parser.add_argument("--methods", type=str, default="base", help="""Method name (this format: "LoRA/QLoRA/QDoRA")""")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into the base model for eval/inference.")
    parser.add_argument("--seed", type=int, default=17, help="Seed number")
    parser.add_argument("--base-path", type=str, default="./", help="Current dir")
    parser.add_argument("--output-path", type=str, default=r"models\<method>_best", help="Checkpoint output path")
    parser.add_argument("--wandb-project", type=str, default="llama-finetune",
                        help="Specify the WandB project name for experiment tracking")
    parser.add_argument("--data-subset", type=int, default=100, help="Subset of the dataset to use")
    parser.add_argument("--use-small-model", help="True to use smaller model (gpt-2) for debugging.")
    return parser.parse_args()


def verify_parsed_args(args, logger):
    # NOT COMPLETE
    # verify argument parsing
    # IT SHOULD LOOK SMTH LIKE THIS:
    # methods = args.methods.lower().strip().split("/")
    # for method in methods:
    #     if method not in allowed:
    #         ...
    if args.methods.lower() not in ['lora', 'qlora', 'dora', 'qdora', 'base']:
        logger.error(f"Invalid method '{args.methods}'. Choose from: lora, qlora, qdora, base.")
        raise ValueError(f"Invalid method '{args.methods}'. Choose from: lora/qlora/qdora/base.")

    if not isinstance(args.seed, int):
        raise ValueError(f"Invalid seed '{args.seed}'. Must be integer.")


def load_and_validate_config(args, logger):
    base_path = args.base_path
    # how should a shared run config + per-method config be loaded here?
    config_path = os.path.join("configs/", f"{args.methods.lower()}_config.yaml")
    logger.debug(f"Config_path: {config_path}.")

    verify_parsed_args(args, logger)

    with open(os.path.join(base_path, config_path), "r") as f:
        config = safe_load(f)
        config["stages"] = args.stages.lower().strip().split("/")
        config["methods"] = args.methods.lower().strip().split("/")
        config["merge"] = args.merge
        config["seed"] = args.seed
        config["base_path"] = args.base_path
        config["data_subset"] = args.data_subset

        config["use_small_model"] = bool(args.use_small_model)

        # config["output_path_template"] = args.output_path
        # config["output_path"] = args.output_path.replace("<method>", config["active_method"])
        config["wandb_project"] = args.wandb_project

    return config


# def check_if_checkpoints_exist(config: Dict, logger):
#     # use only in run_evaluate.py
#     if os.path.exists(config["output_path"]):
#         logger.debug(f"Checkpoints exist at '{config["output_path"]}'")
#     else:
#         config["method"] = "base"
#         logger.warning(f"No checkpoints at '{config["output_path"]}'.\n"
#                        f"Only 'base' method is available. config['method'] set to {config['method']}.")
#         if (input("Do you want to proceed with method set to 'base'? Y/n?")).strip().lower() not in ['y', 'yes']:
#             sys.exit(1)


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