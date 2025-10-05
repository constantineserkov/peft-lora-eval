import warnings

# Ignore this specific FutureWarning from torch.cuda
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.cuda"
)

import torch
from src.logger import set_up_logging, get_logger
from src.utils import set_seed, load_and_validate_config, parse_args

from src.data_loader import (
    load_alpaca_data,
    format_prompt,
    tokenize,
    split_and_sort_dataset,
    get_dataloader, get_tokenized_dataset,
)

from src.auth import init_wandb, init_hf_auth
from src.trainer import train_model
from src.model_utils import configure_peft_model
from src.utils import get_num_warmup_steps, get_num_training_steps
from transformers import AutoTokenizer

logger = get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"

# warn if device is not cuda
if device != "cuda":
    logger.warning("Cuda is not available.")

logger.info(f"Device: {device}")


def main():
    # set up logging baseConfig
    set_up_logging()

    # load config
    args = parse_args()
    config = load_and_validate_config(args)

    # set seed
    set_seed(config["seed"])

    # init wandb and hf
    init_wandb(config)
    init_hf_auth()

    # load dataset
    dataset = load_alpaca_data(
        dataset_name=config['dataset_name'],
        data_subset=config['data_subset']
    )

    # Format dataset -> formatted dataset
    dataset = dataset.map(format_prompt, batched=True, batch_size=100)

    # tokenize dataset
    tokenized_dataset, tokenizer = get_tokenized_dataset(dataset, config)

    # Split the tokenized ds into train/val/test_ds
    datasets = split_and_sort_dataset(
        tokenized_dataset,
        config=config["seed"],
    )
    train_ds, val_ds, _ = datasets.values()

    # debug
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