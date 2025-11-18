import os
from typing import Dict, Any
import contextlib
from tqdm import tqdm
import json
from datetime import datetime
import time
import pynvml

import matplotlib.pyplot as plt
import numpy as np
import wandb

from src.logger import get_logger
from src.trainer import log_vram_usage

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast


logger = get_logger()


def compute_metrics(metrics: Dict) -> Dict:
    """
       Compute and summarize evaluation metrics from raw batch data.

       Parameters
       ----------
       metrics : dict
           A dictionary containing tracked metric lists and values.
           Expected keys:
               - "batches": list or count of processed batches
               - "losses": list of loss values per batch
               - "perplexities": list of perplexity values per batch
               - "vram": list of VRAM usage values per batch

       Returns
       -------
       dict
           A structured dictionary with computed summaries:
               - "batch": raw batch data
               - "evaluation": contains min/max/avg loss and average perplexity
               - "hardware": contains VRAM statistics (avg, peak)
               - "time": reserved fields for timing data
       """
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
            "test_elapsed_time": metrics["test_elapsed"],
            "wall_clock_time": metrics["wc_time"],
        },
    }


def generate_and_save_plots(
        metrics: Dict,
        timestamp: str,
        config: Dict,
        save_to_dir: str = "../results/plots"):
    # Create subdirectories
    path = os.path.join(save_to_dir, f"{config['model']['model_name_or_path']}_{timestamp}")
    evaluation_dir = os.path.join(path, "evaluation")
    hardware_dir = os.path.join(path, "hardware")
    overview_dir = os.path.join(path, "overview")

    for d in [evaluation_dir, hardware_dir, overview_dir]:
        os.makedirs(d, exist_ok=True)
        logger.debug(f"Created/verified {d} dir: {os.path.join(path, d)}")

    # Extract raw lists
    batches = metrics["batches"]  # List of batch indices
    losses = metrics["evaluation"]["losses"]  # List of loss values
    perplexities = metrics["evaluation"]["losses"].get("perplexities", [np.exp(l) for l in losses])  # Fallback to compute from losses if not in metrics
    vram_values = metrics["hardware"]["vram"]  # List of VRAM values in GB

    # evaluation: loss/perplexity curves
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(batches, losses, label="Loss")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss Value")
    plt.title("Loss Curve: Track Convergence/Stability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(evaluation_dir, "loss_vs_batch.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    wandb.log({"loss_curve": wandb.Image(loss_path)})

    # 2 Perplexity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(batches, perplexities, label="Perplixity", color="orange")
    plt.xlabel("Batch Index")
    plt.ylabel("Perplexity Value")
    plt.title("Perplexity Curve: Model Quality Over Evaluation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    perplexity_path = os.path.join(evaluation_dir, "perplexity_vs_batch.png")
    plt.savefig(perplexity_path, dpi=150)
    plt.close()
    wandb.log({"perplexity_curve": wandb.Image(perplexity_path)})

    # 3. Hardware: VRAM Usage
    plt.figure(figsize=(10, 6))
    plt.plot(batches, vram_values, label="VRAM Usage", color="green")
    plt.xlabel("Batch Index")
    plt.ylabel("VRAM (GB)")
    plt.title("VRAM Usage: Memory Profiling")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    vram_batch_path = os.path.join(hardware_dir, "vram_vs_batch.png")
    plt.savefig(vram_batch_path, dpi=150)
    plt.close()
    wandb.log({"vram_usage": wandb.Image(vram_batch_path)})

    # 4. Hardware: VRAM vs Loss (Scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(losses, vram_values, alpha=0.6, color="purple")
    plt.xlabel("Loss Value")
    plt.ylabel("VRAM (GB)")
    plt.title("VRAM vs Loss: Detect Memory Spikes")
    plt.grid(True)
    plt.tight_layout()
    vram_loss_path = os.path.join(hardware_dir, "vram_vs_loss.png")
    plt.savefig(vram_loss_path, dpi=150)
    plt.close()
    wandb.log({"vram_vs_loss": wandb.Image(vram_loss_path)})

    # 5. Overview: Metrics Summary (Bar Chart)
    categories = ["avg_loss", "min_loss", "max_loss", "avg_perplexity", "avg_vram", "peak_vram"]
    values = [
        metrics["evaluation"]["avg_loss"],
        metrics["evaluation"]["min_loss"],
        metrics["evaluation"]["max_loss"],
        metrics["evaluation"]["perplexity"],
        metrics["hardware"]["avg_vram"],
        metrics["hardware"]["peak_vram"]
    ]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=["blue", "green", "red", "orange", "purple", "brown"])
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("Metrics Summary")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', va='bottom')
    summary_path = os.path.join(overview_dir, "metrics_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    wandb.log({"metrics_summary": wandb.Image(summary_path)})

    logger.info(f"Plots generated and saved to {path}")


def save_results(metrics: Dict, metadata: Dict, config: Dict, timestamp) -> None:
    """Save evaluation results to JSON file"""
    results = {
        "metadata": metadata,
        "config": config,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    filename = f"results/metrics/{config["model"]["model_name_or_path"]}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filename}")


def evaluator(
        model: torch.nn.Module,
        dataloader: DataLoader[Dict[str, torch.Tensor]],
        config: Dict,
        device: str,
        metadata: Dict = None,  # Include eval date, dataset split, model ID, seed, and compute (e.g., FLOPs/time). Save as: results/{model_name}_{timestamp}.json.
        tune_hyperparams: bool = False,  #  True to avoid saving results/plots
):
    logger.debug("Running evaluator.")

    # Set up metrics
    temp_metrics: Dict[str, Any] = {
        "batches": [],
        "losses": [],
        "perplexities": [],
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

    # timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if tune_hyperparams:
        return final_metrics["evaluation"]["perplexity"], final_metrics

    # save results
    save_results(final_metrics, metadata, config, timestamp)
    # gen and save plots
    generate_and_save_plots(final_metrics, timestamp, config)

    logger.info(f"Evaluation completed in {temp_metrics['test_elapsed']:.2f}s")
    logger.info(f"Average loss: {final_metrics['evaluation']['avg_loss']:.4f}")
    logger.info(f"Perplexity: {final_metrics['evaluation']['perplexity']:.4f}")

    with contextlib.suppress(Exception):
        wandb.log(final_metrics)

    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlShutdown()
    with contextlib.suppress(Exception):
        wandb.finish()

    return None