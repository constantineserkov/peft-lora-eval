from src.benchmark import run_benchmarks
from src.model_utils import configure_peft_model_for_eval, load_tokenizer

def run_bench(model, method, config, metadata, logger):
    device = metadata["device"]

    model = configure_peft_model_for_eval(model, metadata, config, device, logger)
    tokenizer = load_tokenizer(config, padding_side="left")

    run_benchmarks(model, tokenizer, config, device, logger, metadata)

    return model