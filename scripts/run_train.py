import sys
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

from src.data_loader import unpack_loaders

from src.auth import init_wandb, init_hf_auth
from src.trainer import train_model
from src.model_utils import configure_peft_model_for_training
from src.utils import get_num_warmup_steps, get_num_training_steps


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
    if "google.colab" in sys.modules:
        args.base_path = "/content/drive/MyDrive/peft_lora_eval/"
        logger.info(f"Detected Colab; using base_path: {args.base_path}")
    # add kaggle check

    # run locally
    else:
        logger.info(f"Local run; using base_path: {args.base_path}")
    config = load_and_validate_config(args)

    # set seed
    set_seed(config["seed"])

    # init wandb and hf
    init_wandb(config)
    init_hf_auth()

    train_loader, val_loader, _, tokenizer = unpack_loaders(config)

    # Check what's inside of train_loader
    logger.debug(f"train_loader: {train_loader}")

    # Init model
    model = configure_peft_model_for_training(config_dict=config, device=device)

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