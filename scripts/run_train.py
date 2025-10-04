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
from yaml import safe_load
import os
from src.logger import set_up_logging, get_logger
from src.utils import set_seed, load_and_validate_config

from src.data_loader import (
    load_alpaca_data,
    format_prompt,
    tokenize,
    split_and_sort_dataset,
    get_dataloader,
)

from src.auth import init_wandb, init_hf_auth
from src.trainer import train_model
from src.model_utils import configure_peft_model
from src.utils import get_num_warmup_steps, get_num_training_steps
from transformers import AutoTokenizer

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


def main():
    # set up baseConfig
    set_up_logging()

    # load config
    args = parse_args()
    config = load_and_validate_config(args)
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
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # from transformers import GPT2LMHeadModel  # Add this import at top if needed
        # model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        config['model']['model_name_or_path'] = "gpt2"
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name_or_path'])
        model = configure_peft_model(config_dict=config, device=device)

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
        config_dict=config,
        val_loader=val_loader
    )


if __name__ == "__main__":
    main()