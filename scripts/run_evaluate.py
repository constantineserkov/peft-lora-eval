import time
import torch
from src.auth import init_wandb, init_hf_auth
from src.evaluator import evaluator
from src.logger import set_up_logging, get_logger
from src.model_utils import configure_peft_model_for_eval
from src.utils import parse_args, load_and_validate_config, set_seed, in_colab
from src.data_loader import unpack_loaders


def main():
    # Wall-clock time start
    wc_start = time.time()

    # setup logging base config
    set_up_logging()

    logger = get_logger()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # warn if device is not cuda
    if device != "cuda":
        logger.warning("Cuda is not available.")

    metadata = {
        "wc_start": wc_start,
    }

    # load config + add rewrite parsed args
    args = parse_args()
    if in_colab():
        args.base_path = "/content/drive/MyDrive/peft_lora_eval/"
        logger.info(f"Detected Colab; using base_path: {args.base_path}")
    # add kaggle check

    # run locally
    else:
        logger.info(f"Local run; using base_path: {args.base_path}")
    config = load_and_validate_config(args=args)

    # set seed
    set_seed(config["seed"])

    # init W&B and HF
    init_wandb(config)
    init_hf_auth()

    _, _, test_loader, _ = unpack_loaders(config)

    # Init model
    model = configure_peft_model_for_eval(config_dict=config, device=device)

    # Run evaluation
    evaluator(model, test_loader, config, device, metadata)


if __name__ == "__main__":
    main()