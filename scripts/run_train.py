import logging
import warnings

# Ignore this specific FutureWarning from torch.cuda
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.cuda"
)

import argparse
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
from src.data_loader import (
    load_alpaca_data,
    format_prompt,
    tokenize,
    split_and_sort_dataset,
    get_dataloader,
)
from src.trainer import train_model
from src.model_utils import configure_peft_model
from src.utils import get_num_warmup_steps, get_num_training_steps
from transformers import AutoTokenizer
from huggingface_hub import login as hf_login, HfFolder

logger = get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

# warn if device is not cuda
if device != "cuda":
    logger.warning("Cuda is not available.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 3B fine-tune")

    # add arguments
    parser.add_argument("--mode", type=str, default="test", help="Mode: run/test (run for training, test for testing the pipeline")
    parser.add_argument("--method", type=str, default="lora", help="Method name (LoRA/QLoRA/QDoRA)")
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
    if args.method.lower() not in ['lora', 'qlora', 'dora', 'qdora', 'base']:
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
        config_dict['mode'] = args.mode.lower()
        config_dict['method'] = args.method.lower()
        config_dict['seed'] = args.seed
        config_dict['output_path'] = args.output_path
        config_dict['data_subset'] = args.data_subset

    logging.info(f"Parsed args: \nconfig_path: {config_path}\nmethod: {args.method}\nseed: {args.seed}\n\n"
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


def check_hf_token():
    # 1. Check environment variable
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return True

    # 2. Check Hugging Face token in default folder
    try:
        token = HfFolder.get_token()
        if token:
            logger.info("Hugging Face token is set")
            return True
        else:
            logger.info("No Hugging Face token found")
    except Exception as e:
        logger.error(f"Error checking Hugging Face token: {e}")

    # 3. No token found, prompt user to log in
    logger.info("No Hugging Face token found. Please log in.")
    try:
        hf_login("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")  # temporary
        logger.info("Hugging Face login successful")
        return True
    except Exception:
        logger.error("Hugging Face token not set; skipping model download")
        return False


def init_hf_auth():
    if check_hf_token():
        logger.info("Hugging Face authentication ready.")
    else:
        logger.warning("Proceeding without Hugging Face token. Gated models will be inaccessible.")


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

    # Format dataset -> formatted dataset
    dataset = dataset.map(format_prompt, batched=True, batch_size=100)

    # Hugging Face auth
    init_hf_auth()

    # Init tokenizer and tokenize dataset
    if config['mode'] == "test":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name_or_path'])

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=100,
        fn_kwargs={"tokenizer": tokenizer})

    # Split the tokenized ds into train/val/test_ds
    datasets = split_and_sort_dataset(
        tokenized_dataset,
        seed=config["seed"]
    )
    train_ds, val_ds, test_ds = datasets.values()

    logger.debug(f"Tokenized train dataset sample: {train_ds[0]}\n\n"
                 f"Tokenized val dataset sample: {val_ds[1]}\n\n"
                 f"Tokenized test dataset sample: {test_ds[2]}")

    # Load dataloaders
    train_loader = get_dataloader(
        train_ds,
        tokenizer,
        batch_size=config["dataloader"]["batch_size"],
        seed=config["seed"],
        )
    val_loader = get_dataloader(
        val_ds,
        tokenizer,
        batch_size=config["dataloader"]["batch_size"],
        seed=config["seed"],
    )
    test_loader = get_dataloader(
        test_ds,
        tokenizer,
        batch_size=config["dataloader"]["batch_size"],
        seed=config["seed"],
    )

    # Check what's inside of train_loader
    logger.debug(f"train_loader: {train_loader}")

    # # Wrap into a dict
    # dataloaders = {
    #     "train_loader": train_loader,
    #     "val_loader": val_loader,
    #     "test_loader": test_loader
    # }

    # Init model
    model = configure_peft_model(config_dict=config, device=device)

    # Trainer config setup
    config['training']['num_training_steps'] = get_num_training_steps(train_loader, config)
    config['training']['scheduler']['num_warmup_steps'] = get_num_warmup_steps(
        num_training_steps=config['training']['num_training_steps']
    )

    # Run training
    train_model(
        model, tokenizer,
        train_loader=train_loader,
        device=device,
        onfig_dict=config,
        val_loader=val_loader
    )


if __name__ == "__main__":
    main()