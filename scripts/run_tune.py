import torch.cuda
import wandb
import os
import yaml

import src.tuner as tuner
from src.auth import init_wandb, init_hf_auth
from src.data_loader import unpack_loaders
from src.logger import get_logger, set_up_logging
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

        score, final_metrics = tuner.tune_params(train_loader, test_loader, tokenizer, device, trial_config, config)

        trials.append({'idx': idx, 'config': trial_config, 'score': score})
        wandb.log({'trial_idx': idx, **trial_config, **final_metrics})

    # get best config
    best_conf = tuner.best_config(trials)

    # merge configs
    for k, v in best_conf.items():
        config["peft_config"][k] = v

    # save best config
    config_path = os.path.join("configs/", f"{args.method.lower()}_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    wandb.finish()


if __name__ == "__main__":
    main()