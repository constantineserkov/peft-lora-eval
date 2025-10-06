import time
import torch
from src.auth import init_wandb, init_hf_auth
from src.evaluator import evaluator
from src.logger import set_up_logging, get_logger
from src.model_utils import configure_peft_model_for_eval
from src.utils import parse_args, load_and_validate_config, set_seed
from src.data_loader import (
    load_alpaca_data,
    format_prompt,
    get_tokenized_dataset,
    split_and_sort_dataset,
    get_dataloader
)

# Wall-clock time start
wc_start = time.time()

logger = get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

# warn if device is not cuda
if device != "cuda":
    logger.warning("Cuda is not available.")

metadata = {
    "wc_start": wc_start,

}


def main():
    # setup logging base config
    set_up_logging()

    # load config + add rewrite parsed args
    args = parse_args()
    config = load_and_validate_config(args=args)

    # set seed
    set_seed(config["seed"])

    # init W&B and HF
    init_wandb(config)
    init_hf_auth()

    # load dataset
    dataset = load_alpaca_data(
        dataset_name=config["dataset_name"],
        data_subset=config["data_subset"],
    )

    # format dataset
    formatted_dataset = dataset.map(format_prompt, batched=True, batch_size=100)

    # tokenize dataset
    tokenized_dataset, tokenizer = get_tokenized_dataset(formatted_dataset, config)

    # Split the tokenized ds into train/val/test_ds
    datasets = split_and_sort_dataset(
        tokenized_dataset,
        config=config["seed"],
    )
    _, _, test_ds = datasets.values()

    # debug
    logger.debug(f"Tokenized test dataset sample: {test_ds[17]}")

    # Load test loader
    test_loader = get_dataloader(
        test_ds,
        tokenizer,
        batch_size=config["dataloader"]["batch_size"],
        seed=config["seed"],
    )

    # Init model
    model = configure_peft_model_for_eval(config_dict=config, device=device)

    # Run evaluation
    evaluator(model, test_loader, config, device, metadata)


if __name__ == "__main__":
    main()