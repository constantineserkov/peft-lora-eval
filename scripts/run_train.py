import logging
import warnings

# Ignore this specific FutureWarning from torch.cuda
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.cuda"
)

import argparse
import yaml
import torch
import wandb
from src.data_loader import load_alpaca_data
from src.model_utils import configure_peft_model
from src.trainer import train_model
import sys
from yaml import safe_load
import os
from src.logger import set_up_logging, get_logger
from src.utils import set_seed
from src.data_loader import (load_alpaca_data, format_prompt, tokenize, add_length,
                             get_cleaned_sorted_dataset, get_dataloader)
from transformers import AutoTokenizer

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 3B fine-tune")

    # add arguments
    # parser.add_argument("--config-path", type=str, default=r"configs\base_config.yaml", help="Path to config YAML file")
    parser.add_argument("--method", type=str, default="base", help="Method name (LoRA/QLoRA/QDoRA)")
    parser.add_argument("--seed", type=int, default=17, help="Seed number")
    parser.add_argument("--output-path", type=str, default=r"models\<method>_best", help="Checkpoint output path")
    parser.add_argument("--wandb-project", type=str, default="llama-finetune",
                        help="Specify the WandB project name for experiment tracking")
    parser.add_argument("--data-subset", type=int, default=100, help="Subset of the dataset to use")

    logger.info("Successfully parsed args")

    return parser.parse_args()

def verify_parsed_args(args):
    # NOT COMPLETE
    # verify argument parsing
    if args.method.lower() not in ['lora', 'qlora', 'qdora', 'base']:
        msg = f"Invalid method '{args.method}'. Choose from: lora, qlora, qdora, base."
        logger.error(msg)
        raise ValueError(f"Invalid method '{args.method}'. Choose from: lora/qlora/qdora/base.")

    if not isinstance(args.seed, int):
        raise ValueError(f"Invalid seed '{args.seed}'. Must be integer.")


# load config and verify argument parsing
def load_config(args):
    root_dir = "./"
    config_path = os.path.join("configs/", f"{args.method.lower()}_config.yaml")
    logger.info(f"config_path: {config_path}")

    # verify arguments
    verify_parsed_args(args)

    with open(os.path.join(root_dir, config_path), "r") as f:
        config_dict = safe_load(f)
        config_dict['method'] = args.method
        config_dict['seed'] = args.seed
        config_dict['output_path'] = args.output_path
        config_dict['project_name'] = args.project_name
        config_dict['data_subset'] = args.data_subset

    logging.info(f"Parsed args: \nconfig_path: {config_path}\nmethod: {method}\nseed: {seed}\n\n"
                 f"Config_dict: \n{config_dict}")
    return config_dict


def check_wandb_api_key():
    # 1. Check env var
    if "WANDB_API_KEY" in os.environ:
        return True

    # 2. Check default WandB settings file
    settings_path = os.path.expanduser("~/.config/wandb/settings")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            for line in f:
                if line.startswith("api_key:") and line.split(":", 1)[1].strip():
                    return True

    # 3. If no key found suggest to log in
    logger.info("No WandB API key found. Please log in to WandB.")
    try:
        wandb.login()
        logger.info("WandB login successful")
    except wandb.errors.UsageError:
        logger.error("WandB API key not set; skipping logging")
        return False


def init_wandb(config):
    # W&B initialization if enabled
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        if check_wandb_api_key():
            wandb.init(
                project=config["project_name"],
                config=config
            )
            logger.info("W&B initialized.")
        else:
            logger.warning("WandB API key not set; skipping logging")
    else:
        logger.warning("W&B logging disabled.")


def main():
    # set up baseConfig
    set_up_logging()

    # load config
    args = parse_args()
    config = load_config(args)
    data_subset = config['data_subset']

    # set seed
    set_seed(config['seed'])

    # init wandb
    init_wandb(config)

    # load dataset
    dataset = load_alpaca_data(
        dataset_name=config['dataset_name'],
        data_subset=config['data_subset']
    )

    # format dataset -> formatted dataset
    dataset = dataset.map(format_prompt, batched=True, batch_size=100)

    # init tokenizer and tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name_or_path'])
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=100,
        fn_kwargs={"tokenizer": tokenizer})

    # sort dataset by length
    dataset = tokenized_dataset.map(add_length).map(get_cleaned_sorted_dataset)

    loader = get_dataloader(dataset,
                            tokenizer,
                            batch_size=config['data_loader']['batch_size'],
                            seed=config['seed']
                            )









if __name__ == "__main__":
    main()