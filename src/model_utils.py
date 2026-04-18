import logging
import os
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel, PeftModelForCausalLM,
)

from src.checkpoint import load_checkpoint
from src.utils import select_attn_implementation


def _get_quantization_config(method: str) -> Optional[BitsAndBytesConfig]:
    """Return BitsAndBytes config only for QLoRA/QDoRA."""
    if method not in {"qlora", "qdora"}:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )


def _get_lora_config(
    base_config: Dict,
    use_dora: bool = False,
    from_checkpoint: Optional[Dict] = None,
) -> LoraConfig:
    """Create LoraConfig either from scratch or from checkpoint."""
    if from_checkpoint:
        return from_checkpoint["peft_config"]  # Already a PeftConfig object

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=base_config["r"],
        lora_alpha=base_config["lora_alpha"],
        use_rslora=base_config.get("use_rslora", False),
        target_modules=base_config["target_modules"],
        use_dora=use_dora,
    )


def init_base_model(
        method, config: Dict,
        logger: logging.Logger,
) -> PreTrainedModel:
    # 1. Quantization config (only for QLoRA/QDoRA)
    quantization_config = _get_quantization_config(method)

    # 2. Flash attention implementation
    attn = select_attn_implementation()

    # 3. Load base model
    model_name = config['model']['model_name_or_path']
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
    logger.info("Base model has been loaded successfully.")

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

    # 1. Load checkpoint (if any)
    checkpoint = load_checkpoint(config, metadata, logger)

    # 2. Prepare for k-bit training if needed (only needed for 4-bit)
    quantization_config = _get_quantization_config(method)  # Re-get to check if quantized
    if quantization_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 3. Create or restore PEFT config
    use_dora = method in {"dora", "qdora"}
    peft_config = _get_lora_config(
        base_config=config["peft_config"],
        use_dora=use_dora,
        from_checkpoint=checkpoint,
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
    return model.to(device), checkpoint


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
            adapter_checkpoint_path = os.path.join("runs", metadata["run_id"], method, "checkpoints/best")
            logger.info(f"Loading adapter weights from {adapter_checkpoint_path}")

            model = PeftModelForCausalLM.from_pretrained(model, adapter_checkpoint_path)

            # merge logic (/src/utils.py parse_args)
            if config["merge"]:
                model = model.merge_and_unload()

        except ValueError as e:
            logger.error(f"For method '{method}' no checkpoints were found.")
            raise e

    return model.to(device)