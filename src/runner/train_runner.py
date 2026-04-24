from src.trainer import train_model
from src.model_utils import configure_peft_model_for_training
from src.dataloader import unpack_loaders
from src.utils import get_num_training_steps, get_num_warmup_steps

def run_train(model, method, config, metadata, logger):
    device = metadata["device"]

    if not config.get("training", {}).get("enabled", True):
        logger.info(f"Training disabled for method '{method}'. Skipping train stage.")
        return model

    train_loader, val_loader, _, _ = unpack_loaders(config)
    config["training"]["num_training_steps"] = get_num_training_steps(train_loader, config)
    config["training"]["scheduler"]["num_warmup_steps"] = get_num_warmup_steps(
        config["training"]["num_training_steps"]
    )
    logger.info(
        "Training schedule for method '%s': %s optimizer steps, %s warmup steps.",
        method,
        config["training"]["num_training_steps"],
        config["training"]["scheduler"]["num_warmup_steps"],
    )

    model, checkpoint = configure_peft_model_for_training(model, metadata, config, device, logger)

    train_model(model, device, config, metadata, checkpoint, logger, train_loader, val_loader)

    return model
