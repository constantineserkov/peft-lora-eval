import sys
import os
import torch
from transformers import AutoTokenizer

from src.auth import init_wandb, init_hf_auth
from src.inference import run_inference
from src.logger import get_logger, set_up_logging
from src.model_utils import configure_peft_model_for_eval
from src.utils import parse_args, load_and_validate_config, set_seed, in_colab


def main():
    # setup logging base config
    set_up_logging()

    logger = get_logger()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # warn if device is not cuda
    if device != "cuda":
        logger.warning("Cuda is not available.")

    # load config & rewrite with parsed args
    args = parse_args()
    if in_colab():
        args.base_path = "/content/drive/MyDrive/peft_lora_eval/"
        logger.info(f"Detected Colab; using base_path: {args.base_path}")
    # add kaggle check

    # run locally
    else:
        logger.info(f"Local run; using base_path: {args.base_path}")
    config = load_and_validate_config(args)

    # set seed
    set_seed(config["seed"])

    # auth hf and w&b
    init_wandb(config)
    init_hf_auth()

    # init model, tokenizer
    model = configure_peft_model_for_eval(config, device)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name_or_path"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token


    # run inference
    run_inference(model, tokenizer)


if __name__ == "__main__":
    main()