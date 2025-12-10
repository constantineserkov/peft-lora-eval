import logging
from typing import Dict, List, Set
import os
import torch.nn
from src.runner.common import save_metadata
from peft import get_peft_model_state_dict, PeftModel


def load_checkpoint(
        config, metadata: Dict,
        logger: logging.Logger,
) -> Dict | None:
    """
    Load the latest checkpoint
    Returns
    """
    active_method = config["active_method"]
    latest_checkpoint = metadata.get("latest_checkpoint")
    checkpoint_dir = os.path.join("runs", metadata["run_id"], active_method, "checkpoints/resume")
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Start from scratch if latest checkpoint doesn't exist
    if not latest_checkpoint or not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found. Starting from scratch.")
        clean_checkpoints(checkpoint_dir)
        return None

    return torch.load(checkpoint_path)


def save_best(
        best_loss: float,
        model: PeftModel,
        epoch, step: int,
        config, metadata: Dict,
        logger: logging.Logger,
) -> None:
    """
    Saves best val_loss checkpoints
    """
    active_method = config["active_method"]
    checkpoint_dir = os.path.join("runs", metadata["run_id"], active_method, "checkpoints/best")

    model.save_pretrained(checkpoint_dir)  # saves adapter weights + peft config
    logger.info(f"New best model saved at (epoch {epoch} | step {step}) with eval_loss: \033[1;91m{best_loss:.4f}\033[0m")


def save_checkpoint(
        model: PeftModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        scaler: torch.cuda.amp.GradScaler,
        epoch, step: int,
        best_loss: float,
        config, metadata: Dict,
        logger: logging.Logger,
        final: bool = False,
) -> None:
    """
    Saves resume checkpoints
    """
    active_method = config["active_method"]
    checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pt" if not final else "final.pt"
    checkpoint_dir = os.path.join("runs", metadata["run_id"], active_method, "checkpoints/resume")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "peft_model_state_dict": get_peft_model_state_dict(model),
        "peft_config": model.config,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, checkpoint_path)

    metadata["latest_checkpoint"] = checkpoint_name
    save_metadata(metadata)

    logger.info(f"Checkpoint saved.\nEpoch: {epoch}\nStep: {step}")

    # Optional: clean   old checkpoints
    clean_checkpoints(dir_path=checkpoint_dir, ignore=set(checkpoint_name))


def clean_checkpoints(dir_path, ignore=None) -> None:
    """
    Delete all checkpoints in current dir
    Returns:
    None
    """
    if ignore is None:
        ignore = set()
    if not isinstance(ignore, set):
        ignore = set(ignore)

    for f in os.listdir(dir_path):
        if f in ignore:
            continue
        else:
            fp = os.path.join(dir_path, f)
            if os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                raise f"Current dir {dir_path} supposed to have only files, but dir found!"