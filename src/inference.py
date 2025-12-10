import sys
import time
from itertools import count
import psutil

import numpy as np
import pynvml
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
            "ram_total": 0,  # RAM total (GB)
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
def run_inference(model: torch.nn.Module, tokenizer, metadata):
    start_time = time.time()
    temp_metrics = {
        "latencies_ms": [],
        "input_time": 0,
        "input_token_count": 0,
        "peak_vram":0,
        "vram_usage": [],
        "ram_total": 0,
        "ram_usage": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "energy_consumption": 0,
        "wall_clock_time": 0,

    }

    device = next(model.parameters()).device

    # update first set of metrics
    # get mem info
    mem = psutil.virtual_memory()
    temp_metrics["ram_total"] = mem.total / 1e9  # GB

    # Initialize energy tracking
    prev_time = time.perf_counter()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes single GPU at index 0
    prev_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # kW

    for idx in count(0):
        prompt = get_prompt()

        # exit the program & log metrics
        if prompt == "STOP":
            # wc time
            end_time = time.time()
            temp_metrics["wall_clock_time"] = end_time - start_time

            curr_time = time.perf_counter()
            curr_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # kW
            delta_time = curr_time - prev_time
            avg_power = (prev_power + curr_power) / 2
            temp_metrics["energy_consumption"] += (avg_power * delta_time) / 3600  # Wh

            metrics = compute_metrics(temp_metrics)
            logger.info(f"Inference metrics:\n{metrics}")
            wandb.log({"Inference metrics": metrics})
            logger.info(f"Exiting...")
            sys.exit(1)

        # Start per-prompt timing
        iter_start = time.perf_counter()

        # Tokenization timing
        tok_start = time.perf_counter()
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        tok_end = time.perf_counter()
        input_tokens = len(model_inputs.input_ids[0])
        temp_metrics["input_time"] += (tok_end - tok_start)
        temp_metrics["input_token_count"] += input_tokens  # Per-prompt count
        temp_metrics["total_input_tokens"] += input_tokens

        # Reset VRAM peak for this iteration (if GPU)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Generation
        gen_start = time.perf_counter()
        generated_ids = model.generate(**model_inputs, num_beams=4, do_sample=True)
        gen_end = time.perf_counter()

        # End-to-end latency
        iter_end = time.perf_counter()
        latency_ms = (iter_end - iter_start) * 1000
        temp_metrics["latencies_ms"].append(latency_ms)

        # Output tokens
        output_tokens = len(generated_ids[0]) - input_tokens
        temp_metrics["total_output_tokens"] += output_tokens

        # VRAM tracking (per-prompt peak)
        if torch.cuda.is_available():
            current_vram = torch.cuda.max_memory_allocated(device) / 1e9  # GB
            temp_metrics["vram_usage"].append(current_vram)
            temp_metrics["peak_vram"] = max(temp_metrics["peak_vram"], current_vram)

        # RAM usage (move after generation for full impact)
        mem = psutil.virtual_memory()
        temp_metrics["ram_usage"].append(mem.used / 1e9)  # GB

        # Optional: Energy consumption (delta since last iter; assumes single GPU)
        if 'pynvml' in globals():
            curr_time = time.perf_counter()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            curr_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # kW
            delta_time = curr_time - prev_time
            temp_metrics["energy_consumption"] += (curr_power * delta_time) / 3600  # Wh
            prev_power = curr_power
            prev_time = curr_time

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # log prompt, response
        logger.debug(f"#{idx}:")
        logger.info(f"Prompt: {prompt}\n\n"
                    f"Response: {response}")
        wandb.log({"prompt": prompt, "response": response})


    pynvml.nvmlShutdown()