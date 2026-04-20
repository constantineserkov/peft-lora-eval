from transformers import AutoTokenizer, PreTrainedModel
from src.inference import run_inference
from src.model_utils import configure_peft_model_for_eval

def run_inf(model, method, metadata, logger):
    config = metadata["config"]
    device = metadata["device"]

    config["active_method"] = method

    model = configure_peft_model_for_eval(model, metadata, config, device, logger)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name_or_path"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    run_inference(model, tokenizer, metadata)