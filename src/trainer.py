import logging
from typing import Dict, Optional
import wandb
import pynvml
from peft import PeftModel
from tqdm import tqdm
import gc
import contextlib
import torch
import transformers
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from bitsandbytes.optim import PagedAdamW8bit

from src.metrics import compute_training_metrics, log_vram_usage
from src.checkpoint import save_checkpoint, save_best


def setup_optimizer(
        model: torch.nn.Module,
        config: Dict
) -> Optimizer:
    return PagedAdamW8bit(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=config['training']['optimizer']['betas'],
        eps=config['training']['optimizer']['eps'],
    )


def setup_scheduler(
        optimizer: Optimizer,
        config: Dict
) -> LambdaLR:
    return transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config['training']['scheduler']['num_warmup_steps'],
        num_training_steps=config['training']['num_training_steps']
    )


def train_model(
        model: PeftModel,
        device: str,
        config, metadata, checkpoint: Dict,
        logger: logging.Logger,
        train_loader: DataLoader[Dict[str, torch.Tensor]],
        val_loader: Optional[DataLoader] = None,
) -> None:
    scaler = GradScaler(device)
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)

    log_every = config["logging"]["log_every"]
    g = config["training"]["grad_accumulation_steps"]
    checkpoint_steps = config["logging"]["checkpoint_steps"]

    # Load checkpoint data if checkpoint exists
    if checkpoint:
        start_epoch = checkpoint["epoch"]
        start_step = checkpoint["step"]
        best_loss = checkpoint["best_loss"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(
            "Resuming training from checkpoint at epoch %s, step %s.",
            start_epoch,
            start_step,
        )
    else:
        start_epoch = 1
        start_step = 0
        best_loss: Optional[float] = float("inf")

    # model.train() is set in configure_model_for_training

    # before training: collect garbage and release unused cached memory back to the GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    step: Optional[int] = None
    global_step = 0
    num_epochs = config['training']['num_epochs']

    for epoch in range(start_epoch, num_epochs + 1):
        per_epoch_metrics = {
            "train_losses": [],
            "val_losses": [],
            "perplexity": [],
            "vram_usage": [],
        }
        running_loss = 0
        running_steps = 0

        # Wrap the train loader with tqdm
        pbar = tqdm(train_loader, desc="Training", unit="batch")  # if is off on Jupyter, set position=0
        ran_batches = False

        ### TRAIN ###
        for step, batch in enumerate(pbar):
            if checkpoint and epoch == start_epoch:
                if step <= start_step:
                    continue
            global_step += 1
            ran_batches = True
            batch: Dict[str, torch.Tensor]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.train()
            with autocast(device, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Set Loss description in tqdm
                pbar.set_description(f"Loss: {loss.item():.4f}")

            # Backward pass with scaled loss
            loss_scaled = loss / g
            scaler.scale(loss_scaled).backward()

            running_loss += loss.item()
            running_steps += 1

            # Gradient accumulation
            if (step + 1) % g == 0:
                # Clip grads *after* scaling
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            ##### Log metrics & Validate #####
            if (step + 1) % log_every == 0 or (step + 1) == len(train_loader):
                avg_train_loss = running_loss / max(running_steps, 1)
                per_epoch_metrics["train_losses"].append(avg_train_loss)
                running_loss = 0
                running_steps = 0

                ### Validation ###
                if val_loader:
                    val_running_loss = 0

                    # Wrap the val loader with tqdm
                    pbar_val = tqdm(val_loader, desc="Validation")

                    with torch.no_grad():
                        for val_batch in pbar_val:
                            input_ids = val_batch['input_ids'].to(device)
                            attention_mask = val_batch['attention_mask'].to(device)
                            labels = val_batch['labels'].to(device)

                            model.eval()
                            with autocast(device, dtype=torch.float16):
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                                loss = outputs.loss.item()
                                val_running_loss += loss

                                # Set Loss description in tqdm
                                pbar.set_description(f"Val Loss: {loss:.4f}")

                        ### Metrics & Checkpointing
                        # Compute metrics
                        avg_val_loss = val_running_loss / len(val_loader)
                        metrics = compute_training_metrics(avg_train_loss, avg_val_loss)

                        # Update per epoch metrics
                        per_epoch_metrics["val_losses"].append(avg_val_loss)
                        per_epoch_metrics["perplexity"].append(metrics["perplexity"])
                        per_epoch_metrics["vram_usage"].append(metrics["vram_usage"])

                        ## Save checkpoints
                        # Best loss checkpoint
                        if avg_val_loss < best_loss:  # the lower loss the better
                            best_loss = avg_val_loss
                            save_best(best_loss, model, epoch, step, config, metadata, logger)

                        # Save checkpoint every N steps
                        if (step + 1) % checkpoint_steps == 0:
                            save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_loss, config, metadata, logger)

                        ## Log metrics
                        # Get current learning rate
                        current_lr = scheduler.get_last_lr()[0]  # get_last_lr() -> List[float]
                        with contextlib.suppress(Exception):
                            wandb.log({"lr": current_lr})
                        logger.info(f"Avg. train loss: {metrics["train_loss"]} | "
                                    f"Avg. valid. loss: {metrics["eval_loss"]} | "
                                    f"Perplexity: {metrics["perplexity"]}\n"
                                    f"LR: {current_lr:.1e} | "
                                    f"VRAM usage: {metrics["vram_usage"]} GiB")
                else:
                    logger.info(f"Avg. train loss: {avg_train_loss} | VRAM usage: {log_vram_usage()} GiB")

        # check for a leftover
        logger.debug(f"checking leftover. step == {step}")
        if ran_batches and step is not None and (step + 1) % g != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if checkpoint and epoch == start_epoch and not ran_batches:
            logger.info(
                "Checkpoint already covered epoch %s through step %s. Continuing.",
                epoch,
                start_step,
            )
            if epoch == num_epochs:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch,
                    step, best_loss, config, metadata, logger,
                    final=True
                )
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Save final checkpoint
        if epoch == num_epochs:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                step, best_loss, config, metadata, logger,
                final=True
            )

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

        # Log w&b
        with contextlib.suppress(Exception):
            wandb.log(data=data)

        # Collect garbage and release unused cached memory back to the GPU
        gc.collect()
        torch.cuda.empty_cache()
    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlShutdown()
