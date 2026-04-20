from transformers import AutoTokenizer
from src.benchmark import run_benchmarks
from src.model_utils import configure_peft_model_for_eval

def run_bench(model, method, metadata, logger):
    config = metadata["config"]
    device = metadata["device"]

    model = configure_peft_model_for_eval(model, metadata, config, device, logger)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name_or_path"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    run_benchmarks(model, tokenizer, metadata, config, device)