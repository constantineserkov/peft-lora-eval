import logging
import os
from typing import Dict, Optional, Tuple, cast
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel, PeftModelForCausalLM,
)

from src.checkpoint import load_checkpoint
from src.utils import select_attn_implementation, project_path


def _get_quantization_config(method_quantization_config: Dict) -> Optional[BitsAndBytesConfig]:
    """Return BitsAndBytes config only for QLoRA/QDoRA."""
    if not method_quantization_config.get("load_in_4bit", False):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=method_quantization_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=method_quantization_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_storage=torch.uint8,
    )


def _get_lora_config(
    base_config: Dict,
    use_dora: bool = False,
) -> LoraConfig:
    """Create LoraConfig from the current method config."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=base_config["r"],
        lora_alpha=base_config["lora_alpha"],
        use_rslora=base_config.get("use_rslora", False),
        target_modules=base_config["target_modules"],
        use_dora=use_dora,
    )


def resolve_model_name(config: Dict) -> str:
    """
    Chooses which model name to use according to the config and CLI input --use_small_model
    """
    if config["runtime"].get("use_small_model", False):
        return "gpt2"

    return config["model"]["model_name_or_path"]


def load_tokenizer(
    config: Dict,
    padding_side: str = "left",
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_name(config),
        padding_side=padding_side,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                f"Tokenizer {tokenizer.__class__.__name__} does not define an eos_token. "
                "This project uses eos_token as pad_token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def init_base_model(
        method, config: Dict,
        logger: logging.Logger,
) -> PreTrainedModel:
    # 1. Quantization config (only for QLoRA/QDoRA)
    quantization_config = _get_quantization_config(config["quantization"])
    if quantization_config is None:
        logger.info("Loading base model in non-quantized mode.")
    else:
        logger.info(
            "Loading base model with 4-bit quantization: type=%s, double_quant=%s",
            config["quantization"].get("bnb_4bit_quant_type"),
            config["quantization"].get("bnb_4bit_use_double_quant"),
        )

    # 2. Flash attention implementation
    attn = select_attn_implementation()

    # 3. Load base model
    model_name = resolve_model_name(config)
    logger.debug(f"MODEL NAME: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        offload_folder="offload",
        attn_implementation=attn,
        trust_remote_code=True
    )
    logger.info(
        "Base model loaded successfully for method '%s'. Quantized: %s",
        method,
        quantization_config is not None,
    )
    logger.info(
        "Model quantization flags: is_loaded_in_4bit=%s, is_loaded_in_8bit=%s",
        getattr(model, "is_loaded_in_4bit", False),
        getattr(model, "is_loaded_in_8bit", False),
    )

    return model


def cleanup_model_for_cache(model, config: Dict, logger):
    """
    Return a clean base model suitable for caching.

    If model is PEFT-wrapped, unload adapter weights without merging.
    If unloading is unavailable, return None so caller can evict cache entry.
    """
    if isinstance(model, PeftModel):
        if hasattr(model, "unload"):
            logger.info("Unloading PEFT adapter before returning model to cache.")
            return model.unload()

        logger.warning("PEFT model cannot be unloaded safely; evicting from model cache.")
        return None

    return model


def configure_peft_model_for_training(
        model: PreTrainedModel,
        metadata: Dict,
        config: Dict,
        device: str,
        logger: logging.Logger,
) -> Tuple[PeftModel, Dict]:
    method = config["active_method"]
    logger.info(f"Configuring peft model using method: '{method}'")

    if config.get("peft_config") is None:
        raise ValueError(
            f"Method '{method}' cannot be configured for PEFT training because peft_config is null. "
            "Set training.enabled: false for non-trainable methods, or provide a peft_config."
        )

    # 1. Load checkpoint (if any)
    checkpoint = load_checkpoint(config, metadata, logger)

    # 2. Prepare for k-bit training if needed (only needed for 4-bit)
    quantization_config = _get_quantization_config(config["quantization"])  # Re-get to check if quantized
    if quantization_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 3. Create or restore PEFT config
    use_dora = config["peft_config"].get("use_dora", False)
    peft_config = _get_lora_config(
        base_config=config["peft_config"],
        use_dora=use_dora,
    )

    # 4. Wrap with PEFT
    model = get_peft_model(model, peft_config)

    # 5. Load PEFT weights if checkpoint exists
    if checkpoint:
        model.load_state_dict(checkpoint["peft_model_state_dict"], strict=False)
        logger.info(f"Loaded PEFT weights from checkpoint: {metadata["latest_checkpoint"]}")
    else:
        logger.info("Initialized new PEFT adapter.")

    # 6. Log trainable parameters
    trainable_params, total_params = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {total_params:,} || "
        f"Trainable%: {100 * trainable_params / total_params:.4f}%"
    )

    # 7. Move to device
    model = model.to(device)
    model.train()
    return model, checkpoint


def configure_peft_model_for_eval(
        model: PreTrainedModel,
        metadata: Dict,
        config: Dict,
        device: str,
        logger,
) -> PreTrainedModel | PeftModel | torch.nn.Module:
    method = config["active_method"]
    logger.info(f"Configuring model for evaluation... Using PEFT method: {method}.")

    if method not in {"base", "instruct"}:
        try:
            adapter_checkpoint_path = project_path(config, "runs", metadata["run_id"], method, "checkpoints/best")
            adapter_config_path = os.path.join(adapter_checkpoint_path, "adapter_config.json")
            logger.info(f"Loading adapter weights from {adapter_checkpoint_path}")

            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(
                    f"PEFT adapter checkpoint is missing at '{adapter_checkpoint_path}'. "
                    "Expected adapter_config.json. This usually means training finished without saving a best "
                    "adapter checkpoint."
                )

            model = PeftModelForCausalLM.from_pretrained(model, adapter_checkpoint_path)

            # merge logic (/src/utils.py parse_args)
            if config["runtime"]["merge"]:
                model = model.merge_and_unload()

        except ValueError as e:
            logger.error(f"For method '{method}' no checkpoints were found.")
            raise e

    model = model.to(device)
    model.eval()
    return model
