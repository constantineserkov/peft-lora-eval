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
)

from typing import Dict

from src.logger import get_logger


# get logger
logger = get_logger()


# Load a model and wrap it with peft
def configure_peft_model(config_dict: Dict, device: str) -> torch.nn.Module:
    logger.debug("Configuring a model...")

    # Return a tiny model for testing
    if config_dict["mode"] == "test":
        logger.warning("config_dir['mode'] == 'test': configure_peft_model -> "
                       "tiny model for testing w/o PEFT configuration. "
                       "Model name: 'gpt2'")

        # Create a minimal config for a tiny model
        config = AutoConfig.from_pretrained(
            "gpt2",  # Use a small base model architecture
            vocab_size=10,  # Very small vocab for testing
            n_embd=16,  # Tiny hidden size
            n_layer=1,
            n_head=1,
        )
        return AutoModelForCausalLM.from_config(config).to(device)

    pretrained_model_name_or_path = config_dict['model']['model_name_or_path']

    # If using base method (evaluate the base model)
    if config_dict['method'] in ["base"]:
        logger.info("PEFT is not used. Next step: base model evaluation.")
        return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                low_cpu_mem_usage=True,
                dtype=torch.float16,
                offload_folder="offload",
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            )

    # Log peft method
    logger.info(f"Using PEFT method: ''{config_dict["method"]}'")

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
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
        offload_folder="offload",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    logger.info("Base model has been loaded.")

    # Prepare for k-bit training if needed
    if config_dict["method"] in ["qlora", "qdora"]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )

    # Wrap with PEFT
    model = get_peft_model(
        model,
        peft_config,
    )
    logger.info("Base model has been wrapped with PEFT'")

    # Log trainable parameters once
    logger.info(f"Model trainable parameters: {model.print_trainable_parameters()}")
    return model.to(device)