from src.trainer import train_model
from src.model_utils import configure_peft_model_for_training
from src.dataloader import unpack_loaders

def run_train(model, method, config, metadata, logger):
    device = metadata["device"]

    train_loader, val_loader, _, _ = unpack_loaders(config)
    model, checkpoint = configure_peft_model_for_training(model, metadata, config, device, logger)

    train_model(model, device, config, metadata, checkpoint, logger, train_loader, val_loader)

    return model