from typing import Optional, Dict
import numpy as np
import pynvml
# HOW TO SAVE METRICS AND CHECKPOINTS PATHS
# metadata["artifacts"][method] = {
#     "checkpoint": "checkpoints/lora/final.pt",
#     "train_metrics": "metrics/train_loss.csv",
#     "eval_metrics": "metrics/eval_metrics.json",
#     "plots": {
#         "train": "plots/lora_train_loss.png",
#         "eval": "plots/lora_eval_loss.png"
#     }
# }


def log_vram_usage(device_index: int = 0) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 ** 3)  # GiB


def compute_training_metrics(
        avg_train_loss: float,
        avg_eval_loss: float,
        device_index: Optional[int] = 0,
) -> Dict[str, float]:
    """
    Computes the following metrics:
        1. Avg_train_loss
        2. Avg_eval_loss
        3. Perplexity: (exp(eval_loss))
        4. VRAM usage

    Returns:
        Dict[str, float]: A dictionary with keys:
            - "train_loss": float
            - "eval_loss": float
            - "perplexity": float
            - "vram_usage": float (in MB or GB)
    """
    return {
        "train_loss": avg_train_loss,
        "eval_loss": avg_eval_loss,
        # "perplexity": np.exp(avg_eval_loss), check if this is correct
        "vram_usage": log_vram_usage(device_index)
    }


def compute_eval_metrics(metrics: Dict) -> Dict:
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
            "max_loss": max(metrics["losses"]),
            "min_loss": min(metrics["losses"]),
            "avg_loss": np.mean(metrics["losses"]),
            "perplexity": float(np.exp(np.mean(metrics["losses"]))),
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