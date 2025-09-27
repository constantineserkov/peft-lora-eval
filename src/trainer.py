import logging
import wandb
import torch
import transformers
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def setup_scaler():
    return GradScaler()


def setup_optimizer(model: torch.nn.Module, config_dict: Dict) -> Optimizer:
    """
    Log optimizer settings to WandB (e.g., wandb.log({"optimizer": "AdamW", "lr": config_dict['lr']})).
    Handle errors (e.g., missing config keys) with defaults or validation.
    Add a docstring detailing inputs and output.
    Validate config_dict keys to ensure all required parameters are present.
    Allow flexibility for future optimizers (e.g., Adam for QLoRA/QDoRA if needed).
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=config_dict['lr'],
        weight_decay=config_dict['weight_decay'],
        betas=config_dict['betas'],
        eps=config_dict['eps']
    )


def setup_scheduler(optimizer: Optimizer, config_dict: Dict) -> LambdaLR:
    return transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config_dict['num_warmup_steps'],
        num_training_steps=config_dict['num_training_steps']
    )


def compute_training_metrics() -> Dict:
    return {}


def train_model(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
        config_dict: Dict,
        eval_dataloader: Optional[DataLoader] = None
) -> None:
    scaler = setup_scaler()
    optimizer = setup_optimizer(model, config_dict)
    scheduler = setup_scheduler(optimizer, config_dict)

    for epoch in range(1, config_dict['num_epochs'] + 1):
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
                loss = outputs.loss / config_dict['grad_accumulation_steps']

            # Backward pass with scaled loss
            scaler.scale(loss).backward()

            running_loss += loss.item()

            # gradient accumulation
            if step % config_dict['grad_accumulation_steps'] == 0:
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
        if (step + 1) % config_dict['grad_accumulation_steps'] != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if eval_dataloader:
            metrics = compute_training_metrics()

        epoch_loss = sum(train_losses) / len(train_losses)
        print(f"\n\nEpoch {epoch} average loss: {epoch_loss}\n\n")

        # release unused cached memory back to the GPU
        torch.cuda.empty_cache()


def log_vram_usage(device: Optional[str]) -> float:
    pass


# save model's checkpoints to disk each N num_steps
def save_checkpoint(num_steps: int) -> None:
    pass