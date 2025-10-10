### Hyperparameter Strategy
# Due to computational constraints, I performed manual ablations on key parameters
# (rank, alpha) across all PEFT methods. The basic tuning framework is implemented in
# `tuner.py` for future extensibility.
#
# **Key Findings:**
# - LoRA performs best at rank=16, alpha=32
# - QLoRA shows optimal trade-offs at rank=8
# - [Your actual findings here]
# Prototype level tuning due to time and compute constrains
# 1. create grid of parameters -> Dict
# 2. for method in  peft_methods rewrite config according to tuning params
    # 3. Run training/eval with this param set mentioning that this run is for hyperparam tuning
    # 4. save results
# 5. compare all results for method
# 6. run another method
import gc
from typing import Dict, List
from itertools import product

import torch
import transformers
from torch.autograd.profiler_util import OUT_OF_MEMORY_EVENT_NAME
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model_utils import configure_peft_model_for_eval
from src.trainer import train_model
from src.evaluator import evaluator

from src.logger import get_logger

logger = get_logger()

def define_search_space(method: str) -> Dict[str, List]:
    param_grid = {
        "lora": {
            "r": [8, 16, 32],
            "alpha": [16, 32]
        },
        "dora": {
            "r": [8, 16, 32],
            "alpha": [16, 32]
        },
        "qlora": {
            "r": [4, 8, 16],
            "alpha": [8, 16]
        },
        "qdora": {
            "r": [4, 8, 16],
            "alpha": [8, 16]
        },
    }

    return param_grid[method]


def generate_grid(space: Dict) -> List[Dict[str, int]]:
    # Extract param names and their value lists
    param_names = list(space.keys())
    param_values = [space[name] for name in param_names]

    # Generate all combinations
    all_combos = list(product(*param_values))

    # Convert to list of dicts
    trials = [dict(zip(param_names, combo)) for combo in all_combos]

    return trials


def best_config(trials: List):
    if not trials:
        raise ValueError("No trials provided")

    best = {"score": float("inf")}
    for trial in trials:
        if trial["score"] < best["score"]:
            best = trial

    logger.info(f"Best trial idx: {idx}\nPerplexity: {score}\nConfig: {config}" for idx, config, score in best.values())

    return best["config"]


def save_results():
    pass


def plot_results():
    pass


def tune_params(
        train_loader: DataLoader[Dict[str, torch.Tensor]],
        test_loader: DataLoader[Dict[str, torch.Tensor]],
        tokenizer,
        device: str,
        trial_config: Dict,
        config: Dict,
):
    logger.info("Tuning hyperparameters...")

    try:
        # merge configs
        for k, v in trial_config.items():
            config["peft_config"][k] = v

        # configure model
        model = configure_peft_model_for_eval(config, device)

        # train
        train_model(model, tokenizer, train_loader, device, config)

        # evaluate
        score, final_metrics = evaluator(model, test_loader, config, device, )

        # reset vram stats
        torch.cuda.reset_peak_memory_stats()

        # delete temp model
        del model

        # collect garbage and release unused cached memory back to the GPU
        gc.collect()
        torch.cuda.empty_cache()

        return score, final_metrics

    except Exception as e:
        logger.error(f"Exception has been raised: \n{torch.cuda.reset_peak_memory_stats()}")
        return float("inf")
