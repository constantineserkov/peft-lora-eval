import time
from typing import Dict, Optional
from src.logger import get_logger
import wandb
import pynvml
from tqdm import tqdm
import gc
import contextlib

import torch
import numpy as np
import transformers
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from bitsandbytes.optim import PagedAdamW8bit

logger = get_logger()


def setup_scaler(device: str):
    return GradScaler(device)


def setup_optimizer(model: torch.nn.Module, config_dict: Dict) -> Optimizer:
    # return torch.optim.AdamW(
    #     model.parameters(),
    #     lr=config_dict['training']['lr'],
    #     weight_decay=config_dict['training']['optimizer']['weight_decay'],
    #     betas=config_dict['training']['optimizer']['betas'],
    #     eps=config_dict['training']['optimizer']['eps']
    # )
    return PagedAdamW8bit(
        model.parameters(),
        lr=config_dict['training']['lr'],
        weight_decay=config_dict['training']['optimizer']['weight_decay'],
        betas=config_dict['training']['optimizer']['betas'],
        eps=config_dict['training']['optimizer']['eps'],
    )



def setup_scheduler(optimizer: Optimizer, config_dict: Dict) -> LambdaLR:
    return transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config_dict['training']['scheduler']['num_warmup_steps'],
        num_training_steps=config_dict['training']['num_training_steps']
    )


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
        "perplexity": np.exp(avg_eval_loss),
        "vram_usage": log_vram_usage(device_index)
    }


# Log VRAM with pynvml
def log_vram_usage(device_index: int = 0) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_gb = mem_info.used / (1024 ** 3)  # GB
    return used_gb


# Save model's checkpoints to disk each N num_steps
def save_checkpoint(avg_eval_loss, model, tokenizer, epoch, config_dict) -> float:
    best_loss = avg_eval_loss
    model.save_pretrained(config_dict["output_path"])
    tokenizer.save_pretrained(config_dict["output_path"])
    logger(f"New best model saved at epoch {epoch} with eval_loss: {avg_eval_loss:.4f}")
    return best_loss

def train_model(
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        train_loader: DataLoader[Dict[str, torch.Tensor]],
        device: str,
        config_dict: Dict,
        val_loader: Optional[DataLoader] = None,
        tune_hyperparams: bool = False,  # if hyperparam tuning, do not save checkpoints
) -> None:

    logger.info(f"LR type: {type(config_dict['training']['lr'])}")
    scaler = setup_scaler(device)
    optimizer = setup_optimizer(model, config_dict)
    scheduler = setup_scheduler(optimizer, config_dict)

    best_loss = float("inf")
    model.train()
    # IMPLEMENT time tracking train_time = tot_time - val_time
    # start_time = time.time()

    # before training: collect garbage and release unused cached memory back to the GPU
    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(1, config_dict['training']['num_epochs'] + 1):
        per_epoch_metrics = {
            "train_losses": [],
            "val_losses": [],
            "perplexity": [],
            "vram_usage": [],
        }
        log_every = 500
        running_loss = 0

        # Wrap the train loader with tqdm
        pbar = tqdm(train_loader, desc="Training", unit="batch")  # if is off on Jupyter, set position=0

        step = None
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.train()
            with autocast(device, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config_dict['training']['grad_accumulation_steps']

                # set Loss description in tqdm
                pbar.set_description(f"Loss: {loss.item():.4f}")

            # Backward pass with scaled loss
            scaler.scale(loss).backward()

            running_loss += loss.item()

            # gradient accumulation
            if step % config_dict['training']['grad_accumulation_steps'] == 0:
                # clip grads *after* scaling
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % log_every == 0:
                avg_train_loss = running_loss / log_every
                per_epoch_metrics["train_losses"].append(avg_train_loss)
                running_loss = 0

                # Validation
                if val_loader:
                    val_running_loss = 0

                    # Wrap the val loader with tqdm
                    pbar_val = tqdm(val_loader, desc="Validation")

                    with torch.no_grad():  # Disable gradients
                        for val_batch in pbar_val:
                            input_ids = val_batch['input_ids'].to(device)
                            attention_mask = val_batch['attention_mask'].to(device)
                            labels = val_batch['labels'].to(device)

                            model.eval()
                            with autocast(device, dtype=torch.float16):
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                                loss = outputs.loss.item()
                                val_running_loss += loss

                                # set Loss desctiption in tqdm
                                pbar.set_description(f"Val Loss: {loss:.4f}")

                        # Compute metrics
                        avg_eval_loss = val_running_loss / len(val_loader)
                        metrics = compute_training_metrics(avg_train_loss, avg_eval_loss)

                        # Update per epoch metrics
                        per_epoch_metrics["val_losses"].append(avg_eval_loss)
                        per_epoch_metrics["vram_usage"].append(metrics["vram_usage"])

                        # Checkpoint logic: save if best; update best_loss
                        if avg_eval_loss < best_loss:
                            best_loss = save_checkpoint(avg_eval_loss, best_loss, model, tokenizer, epoch, config_dict)

                        # Log metrics
                        current_lr = scheduler.get_last_lr()[0]  # get_last_lr() -> List[float]
                        wandb.log({"lr": current_lr})
                        logger.info(f"Step: {step}\n"
                                    f"Avg. train loss: {metrics["train_loss"]}\n"
                                    f"Avg. valid. loss: {metrics["eval_loss"]}\n"
                                    f"Perplexity: {metrics["perplexity"]}\n"
                                    f"LR: {current_lr:.1e}"
                                    f"VRAM usage: {metrics["vram_usage"]} GB")
                else:
                    logger.info(f"Avg. train loss: {avg_train_loss}\n"
                                f"VRAM usage: {log_vram_usage()} GB")

        # check for a leftover
        logger.debug(f"checking leftover. step == {step}")
        if (step + 1) % config_dict['training']['grad_accumulation_steps'] != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # check if lists are empty
        avg_train_epoch_loss = sum(per_epoch_metrics["train_losses"]) / len(per_epoch_metrics["train_losses"]) if (
            per_epoch_metrics)["train_losses"] else 0
        avg_val_epoch_loss = sum(per_epoch_metrics["val_losses"]) / len(per_epoch_metrics["val_losses"]) if (
            per_epoch_metrics)["val_losses"] else 0
        avg_perplexity = sum(per_epoch_metrics["perplexity"]) / len(per_epoch_metrics["perplexity"]) if (
            per_epoch_metrics)["perplexity"] else 0
        avg_vram_usage = sum(per_epoch_metrics["vram_usage"]) / len(per_epoch_metrics["vram_usage"]) if (
            per_epoch_metrics)["vram_usage"] else 0

        # populate logging data dict
        data = {
            "epoch": epoch,
            "avg_train_epoch_loss": avg_train_epoch_loss,
            "avg_val_epoch_loss": avg_val_epoch_loss,
            "avg_perplexity": avg_perplexity,
            "avg_vram_usage": avg_vram_usage,
        }
        with contextlib.suppress(Exception):
            wandb.log(data=data)

        # collect garbage and release unused cached memory back to the GPU
        gc.collect()
        torch.cuda.empty_cache()

    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlShutdown()

    with contextlib.suppress(Exception):
        wandb.finish()