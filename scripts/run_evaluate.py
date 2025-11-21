import time
import torch
from src.auth import init_wandb, init_hf_auth
from src.evaluator import evaluator
from src.logger import set_up_logging, get_logger
from src.model_utils import configure_peft_model_for_eval
from src.data_loader import unpack_loaders

from src.utils import (
    parse_args,
    load_and_validate_config,
    set_seed,
    resolve_device,
    resolve_base_path,
)


def main():
    wc_start = time.time()

    set_up_logging()
    logger = get_logger()

    device = resolve_device()
    logger.info(f"Device: {device}")
    if device != "cuda":
        logger.warning("Cuda is not available.")

    args = parse_args()
    args.base_path = resolve_base_path(args)

    config = load_and_validate_config(args=args)
    set_seed(config["seed"])

    init_wandb(config)
    init_hf_auth()

    _, _, test_loader, _ = unpack_loaders(config)
    model = configure_peft_model_for_eval(config_dict=config, device=device)

    metadata = {"wc_start": wc_start}

    evaluator(model, test_loader, config, device, metadata)


if __name__ == "__main__":
    main()