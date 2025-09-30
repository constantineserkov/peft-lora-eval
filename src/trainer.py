import logging
import wandb
import torch
import numpy as np
import transformers
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from src.logger import get_logger
from bitsandbytes.optim import PagedAdamW8bit
import pynvml

logger = get_logger()


def setup_scaler():
    return GradScaler()


def setup_optimizer(model: torch.nn.Module, config_dict: Dict) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config_dict['training']['lr'],
        weight_decay=config_dict['training']['optimizer']['weight_decay'],
        betas=config_dict['training']['optimizer']['betas'],
        eps=config_dict['training']['optimizer']['eps']
    )
    # return PagedAdamW8bit(
    #     model.parameters(),
    #     lr=config_dict['training']['lr'],
    #     weight_decay=config_dict['training']['optimizer']['weight_decay'],
    #     betas=config_dict['training']['optimizer']['betas'],
    #     eps=config_dict['training']['optimizer']['eps'],
    # )



def setup_scheduler(optimizer: Optimizer, config_dict: Dict) -> LambdaLR:
    return transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config_dict['training']['scheduler']['num_warmup_steps'],
        num_training_steps=config_dict['training']['num_training_steps']
    )


def compute_training_metrics(avg_train_loss: float, avg_eval_loss: float, device_index: int) -> Dict[str, float]:
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


def train_model(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
        config_dict: Dict,
        eval_dataloader: Optional[DataLoader] = None
) -> None:

    logger.info(f"LR type: {type(config_dict['training']['lr'])}")
    scaler = setup_scaler()
    optimizer = setup_optimizer(model, config_dict)
    scheduler = setup_scheduler(optimizer, config_dict)

    for epoch in range(1, config_dict['training']['num_epochs'] + 1):
        train_losses = []
        log_every = 500
        running_loss = 0

        model.train()
        step = None
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config_dict['training']['grad_accumulation_steps']

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
                print(f"Step {step + 1}: average loss = {running_loss / log_every}")
                running_loss = 0

        # check for a leftover
        if (step + 1) % config_dict['training']['grad_accumulation_steps'] != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if eval_dataloader:
            # create two val loaders one small and one bigger for every epoch and every 1k steps validation
            metrics = compute_training_metrics(avg_train_loss, avg_eval_loss)

        epoch_loss = sum(train_losses) / len(train_losses)
        print(f"\n\nEpoch {epoch} average loss: {epoch_loss}\n\n")

        # release unused cached memory back to the GPU
        torch.cuda.empty_cache()
        pynvml.nvmlShutdown()


def log_vram_usage(device_index: int = 0) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info / (1024 ** 2)  # MB


# save model's checkpoints to disk each N num_steps
def save_checkpoint(num_steps: int) -> None:
    pass