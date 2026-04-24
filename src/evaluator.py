from typing import Dict
import contextlib
from tqdm import tqdm
from datetime import datetime
import time
import pynvml
import wandb
from src.results import save_results, generate_and_save_plots
from src.trainer import log_vram_usage
from src.metrics import compute_eval_metrics, measure_runtime_and_peak_vram
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast


def evaluator(
        model: torch.nn.Module,
        dataloader: DataLoader[Dict[str, torch.Tensor]],
        config: Dict,
        device: str,
        logger,
        metadata: Dict = None,  # Include eval date, dataset split, model ID, seed, and compute (e.g., FLOPs/time). Save as: results/{model_name}_{timestamp}.json.
        tune_hyperparams: bool = False,  #  True to avoid saving results/plots
):
    ### LOG EVERY BATCH (STEP)
    logger.debug("Running evaluator.")
    metadata["metric_type"] = "evaluator"
    temp_metrics: Dict[str, list[float] | list[int] | float | None] = {
        "batches": [],
        "losses": [],
        "vram": [],
        "test_elapsed": None,
        "wc_time": None,
    }

    # model.eval() is set in configure_model_for_eval
    wall_clock_start = time.time()
    with measure_runtime_and_peak_vram(device) as runtime_metrics:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluation")
            for batch_idx, batch in enumerate(pbar):
                batch: Dict[str, torch.Tensor]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast(device, dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss.item()

                    # update metrics
                    temp_metrics["batches"].append(batch_idx)
                    temp_metrics["losses"].append(loss)
                    temp_metrics["vram"].append(log_vram_usage())

    wall_clock_end = time.time()

    temp_metrics["test_elapsed"] = runtime_metrics["elapsed_seconds"]
    temp_metrics["wc_time"] = wall_clock_end - metadata.get("wc_time", wall_clock_start)  # fallback if wc_time is missing
    final_metrics = compute_eval_metrics(temp_metrics)

    final_metrics["hardware"]["peak_vram_torch_gb"] = runtime_metrics["peak_vram_torch_gb"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.debug("Saving results...")
    save_results(final_metrics, metadata, config, logger, timestamp)

    logger.debug("Generating and saving plots...")
    generate_and_save_plots(final_metrics, timestamp, config, logger, "TEMP", metadata)

    logger.info(f"Evaluation completed in {temp_metrics['test_elapsed']:.2f}s")
    logger.info(f"Average loss: {final_metrics['evaluation']['avg_loss']:.4f}")
    logger.info(f"Perplexity: {final_metrics['evaluation']['perplexity']:.4f}")
    logger.info(f"Final metrics: {final_metrics}")

    with contextlib.suppress(Exception):
        wandb.log(final_metrics)
    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlShutdown()
    with contextlib.suppress(Exception):
        wandb.finish()

    return None
