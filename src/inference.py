import sys
from typing import Dict
from itertools import count
import psutil

import numpy as np
import torch
import wandb

from src.logger import get_logger


logger = get_logger()


def compute_metrics(metrics):
    logger.debug("Computing metrics...")
    return {
        "prompts": [],  # list of prompts indices
        # Latency & Throughput (per-prompt and aggregates)
        "inference_latency": {
            "tpot": 0,  # Time Per Output Token
            "tps": 0,  # Tokens Per Second
            "p50_latency": np.median(metrics["latencies_ms"]),  # Median end-to-end latency (ms) for robustness to outliers
            "p95_latency": np.percentile(metrics["latencies_ms"], 95),  # 95th percentile latency (ms) – critical for real-time apps
            "avg_tpit": 0,  # Time Per Input Token (ms) – measures preprocessing overhead
        },
        # Memory Usage (per-prompt and aggregates)
        "memory_usage": {
            "vram_usage": metrics["vram_usage"],
            "peak_vram": metrics["peak_vram"],  # Global max VRAM across all prompts
            "avg_vram": np.mean(metrics["vram_usage"]),  # Average VRAM usage
            "ram_available": 0,  # RAM available (GB)
            "ram_usage": metrics["ram_usage"],  # List of peak system RAM (GB) per prompt – for holistic resource view
        },
        # Timing & Compute (global)
        "time_and_compute": {
            "wall_clock_time": 0,  # Total wall-clock time for all prompts
            "avg_time": 0,  # Average prompt time
            "compute_flops": 0,  # Total FLOPs
            "avg_flops": 0,  # Average FLOPs
            "energy_consumption": 0,  # Total energy (Wh) if measurable – for sustainability
        },
    }


# get input prompt
def get_prompt() -> str:
    return input("Enter your prompt: ")


# run inference
def run_inference(model: torch.nn.Module, tokenizer):
    temp_metrics = {
        "latencies_ms": [],
        "input_time": 0,
        "input_token_count": 0,
        "peak_vram":0,
        "vram_usage": [],
        "ram_usage": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "energy_consumption": 0,
    }

    for idx in count(0):
        prompt = get_prompt()

        # return STOP to exit the program
        if prompt == "STOP":
            logger.info(f"Inference metrics:\n{compute_metrics(temp_metrics)}")
            logger.info(f"Exiting...")
            sys.exit(1)

        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, num_beams=4, do_sample=True)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # metrics


        # log prompt, response
        logger.debug(f"#{idx}:")
        logger.info(f"Prompt: {prompt}\n\n"
                    f"Response: {response}")
        wandb.log({"prompt": prompt, "response": response})