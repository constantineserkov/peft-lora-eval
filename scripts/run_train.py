import argparse
import yaml
import torch
import wandb
import logging
from src.data_loader import load_alpaca_data
from src.model_utils import configure_peft_model
from src.trainer import train_model
import sys


def main() -> None:
    """
    Colab: The default (/content/llama-alpaca-finetune/models/lora_best) is temporary and lost after session ends.
    To persist, users can override with --output_dir /content/drive/MyDrive/llama_checkpoints/lora
    (requires Drive mounting).
    """
    parser = argparse.ArgumentParser(description="Run Llama 3.2 3B fine-tune")

    # add arguments
    parser.add_argument("--config-path", type=str, default=r"configs\base_config.yaml", help="Path to config YAML file")
    parser.add_argument("--method", type=str, default="lora", help="Method name (LoRA/QLoRA/QDoRA)")
    parser.add_argument("--seed", type=int, default=17, help="Seed number")
    parser.add_argument("--output-path", type=str, default=r"models\<method>_best", help="Checkpoint output path")
    parser.add_argument("--wandb-project", type=str, default="llama-finetune",
                        help="Specify the WandB project name for experiment tracking")
    parser.add_argument("--data-subset", type=int, default=100, help="Subset of the dataset to use")

    # parse arguments
    args = parser.parse_args()

    # configure config path, peft method, seed
    config_path = args.config_path
    method = args.method
    seed = args.seed

    # verify argument parsing
    if args.method.lower() not in ['lora', 'qlora', 'qdora', 'base']:
        pass  # warning








    if __name__ == "__main__":
        main()