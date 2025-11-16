import torch

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModelForCausalLM,
)

from typing import Dict

from src.logger import get_logger


# get logger
logger = get_logger()


# Load a model and wrap it with peft
def configure_peft_model_for_training(
        config_dict: Dict,
        device: str
) -> torch.nn.Module:
    # Log peft method
    logger.info(
        f"Configuring a model... Using PEFT method: '{config_dict["method"]}'"
    )

    # Init PEFT config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config_dict['peft_config']['r'],
        lora_alpha=config_dict['peft_config']['lora_alpha'],
        use_rslora=config_dict['peft_config']['use_rslora'],
        target_modules=config_dict['peft_config']['target_modules']
    )

    # Determine PEFT flags
    if config_dict["method"] in ["dora", "qdora"]:
        peft_config.use_dora = True

    # Determine quantization config
    bnb_config = None
    if config_dict["method"] in ["qlora", "qdora"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage="uint8",
        )

    # Load model
    logger.debug("MODEL NAME:", config_dict['model']['model_name_or_path'])
    model = AutoModelForCausalLM.from_pretrained(
        config_dict['model']['model_name_or_path'],
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        offload_folder="offload",
        # attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        attn_implementation="eager",
    )
    logger.info("Base model has been loaded.")
    logger.debug(f"Model modules names: {model.named_modules()}")

    # Prepare for k-bit training if needed
    if config_dict["method"] in ["qlora", "qdora"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Wrap with PEFT
    model = get_peft_model(model, peft_config)

    # Log trainable parameters once
    logger.info(f"Model trainable parameters: {model.print_trainable_parameters()}")

    return model.to(device)


def configure_peft_model_for_eval(
        config_dict: Dict,
        device: str,
) -> torch.nn.Module:
    logger.debug("Loading model for evaluation...")

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        config_dict["model"]["model_name_or_path"],
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        offload_folder="offload",
        # attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        attn_implementation="eager",
    )

    # if it's a peft method
    if config_dict["method"] != "base":
        adapter_checkpoint_path = config_dict["output_path"]
        logger.info(f"Loading adapter weights from {adapter_checkpoint_path}")

        model = PeftModelForCausalLM.from_pretrained(model, adapter_checkpoint_path)

        # merge logic if merge set True
        if config_dict["merge"]:
            model = model.merge_and_unload()

    return model.to(device)