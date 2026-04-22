from src.evaluator import evaluator
from src.model_utils import configure_peft_model_for_eval
from src.dataloader import unpack_loaders

def run_eval(model, method, config, metadata, logger):
    device = metadata["device"]

    _, _, test_loader, _ = unpack_loaders(config)
    model = configure_peft_model_for_eval(model, metadata, config, device, logger)

    evaluator(model, test_loader, config, device, logger, metadata)

    return model