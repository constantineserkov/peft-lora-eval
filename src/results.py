import os
from typing import Dict
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import wandb


def generate_and_save_plots(
        metrics: Dict,
        timestamp: str,
        config: Dict,
        logger,
        save_to_dir,
        metadata,
):
    # Create subdirectories
    path = os.path.join(save_to_dir, f"{config['model']['model_name_or_path']}_{timestamp}")
    evaluation_dir = os.path.join(path, "evaluation")
    hardware_dir = os.path.join(path, "hardware")
    overview_dir = os.path.join(path, "overview")

    for d in [evaluation_dir, hardware_dir, overview_dir]:
        os.makedirs(d, exist_ok=True)
        logger.debug(f"Created/verified {d} dir: {os.path.join(path, d)}")

    # Extract raw lists
    batches = metrics["batch"]  # List of batch indices
    losses = metrics["evaluation"]["losses"]
    perplexities = [np.exp(l) for l in losses]
    vram_values = metrics["hardware"]["vram"]

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
    plt.plot(batches, perplexities, label="Perple1kxity", color="orange")
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


def save_results(
        metrics: Dict,
        metadata: Dict | None,
        config: Dict,
        logger,
        timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
) -> None:
    """Save evaluation results to JSON file"""
    results = {
        "metadata": metadata,
        "config": config,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    # Build path
    model_name = config["model"]["model_name_or_path"]
    safe_model_name = model_name.replace('/', '_')  # replace '/' to avoid unnecessary folder creation
    peft_method_name = config["active_method"]

    if metadata["metric_type"] == "evaluator":
        save_dir = os.path.join("results", "metrics")
    elif metadata["metric_type"] == "benchmark":
        save_dir = os.path.join("results", "benchmarks")
    else:
        logger.warning(f"Unusual metric_type: {metadata["metric_type"]}. Should be either 'evaluator', or 'benchmark'")
        save_dir = os.path.join("results", input("Where do you want to save the metrics? Current dir: 'results/'. "
                                                 "Input(metrics/benchmarks): ").lower())
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"{safe_model_name}_{peft_method_name}_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filename}.")