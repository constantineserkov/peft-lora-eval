# What is liger kernel for qlora
from typing import Dict, Any

import numpy as np
import wandb
from tqdm import tqdm

from src.logger import get_logger
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import json
from datetime import datetime
import time
import pynvml

from trainer import log_vram_usage


logger = get_logger()


def compute_metrics(metrics: Dict) -> Dict:
    return {
        "batch": metrics["batches"],

        "evaluation": {
            "losses": metrics["losses"],
            "max_loss": max(metrics["losses"]) if metrics["losses"] else 0,
            "min_loss": min(metrics["losses"]) if metrics["losses"] else 0,
            "avg_loss": sum(metrics["losses"]) / len(metrics["losses"]) if metrics["losses"] else 0,
            "perplexity": sum(metrics["perplexities"]) / len(metrics["perplexities"]) if metrics["perplexities"] else 0,
        },
        "hardware": {
            "vram": metrics["vram"],
            "avg_vram": sum(metrics["vram"]) / len(metrics["vram"]) if metrics["vram"] else 0,
            "peak_vram": max(metrics["vram"]) if metrics["vram"] else 0,
        },
        "time": {
            "test_elapsed_time": None,
            "wall_clock_time": None,
        },
    }


def save_results(metrics: Dict, metadata: Dict, config: Dict) -> None:
    """Save evaluation results to JSON file"""
    results = {
        "metadata": metadata,
        "config": config,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    filename = f"results/metrics/{config["model_name"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filename}")


def evaluator(
        model: torch.nn.Module,
        dataloader: DataLoader[Dict[str, torch.Tensor]],
        config: Dict,
        device: str,
        metadata: Dict,  # Include eval date, dataset split, model ID, seed, and compute (e.g., FLOPs/time). Save as: results/{model_name}_{timestamp}.json.
) -> None:
    logger.debug("Running evaluator.")

    # Set up metrics
    temp_metrics: Dict[str, Any] = {
        "batches": [],
        "losses": [],
        "perplexity": [],
        "vram": [],
        "test_elapsed": None,
        "wc_time": None,
    }

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        # Wrap the val loader with tqdm
        pbar = tqdm(dataloader, desc="Evaluation")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.item()

                # update metrics
                temp_metrics["batches"].append(batch_idx)
                temp_metrics["losses"].append(loss)
                temp_metrics["perplexities"].append(np.exp(loss))
                temp_metrics["vram"].append(log_vram_usage())

    end_time = time.time()

    temp_metrics["test_elapsed"] = end_time - start_time
    temp_metrics["wc_time"] = end_time - metadata["wc_start"]

    # compute final metrics
    final_metrics = compute_metrics(temp_metrics)

    # save results
    save_results(final_metrics, metadata, config)

    logger.info(f"Evaluation completed in {temp_metrics['test_elapsed']:.2f}s")
    logger.info(f"Average loss: {final_metrics['evaluation']['avg_loss']:.4f}")
    logger.info(f"Perplexity: {final_metrics['evaluation']['perplexity']:.4f}")

    wandb.log(final_metrics)

    try:
        pynvml.nvmlShutdown()
    except:
        pass
    wandb.finish()