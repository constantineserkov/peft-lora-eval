import torch.cuda
import wandb

import src.tuner as tuner
from src.auth import init_wandb, init_hf_auth
from src.data_loader import load_alpaca_data, format_prompt, get_tokenized_dataset, split_and_sort_dataset, \
    get_dataloader, unpack_loaders
from src.logger import get_logger, set_up_logging
from src.tuner import best_config
from src.utils import load_and_validate_config, parse_args, set_seed

logger = get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"

# warn if device is not cuda
if device != "cuda":
    logger.warning("Cuda is not available.")

logger.info(f"Device: {device}")


def main():
    set_up_logging()

    # parse args
    args = parse_args()
    config = load_and_validate_config(args)

    # set seed
    set_seed(config["seed"])

    # init
    init_wandb(config)
    init_hf_auth()

    # setup loaders, etc
    train_loader, _, test_loader, tokenizer = unpack_loaders(config)

    # tuning loop logic
    search_space = tuner.define_search_space(config["method"])
    grid = tuner.generate_grid(search_space)

    # list to collect trials
    trials = []

    for idx, trial_config in enumerate(grid):
        logger.info(f"Running trial {idx+1}/{len(grid)} with config {trial_config}")

        score, final_metrics = tuner.tune_params(train_loader, test_loader, tokenizer, device=device, )

        trials.append({'idx': idx, 'config': trial_config, 'score': score})
        wandb.log({'trial_idx': idx, **trial_config, **final_metrics})

    best_conf = tuner.best_config(trials)

    # save best config


if __name__ == "__main__":
    main()