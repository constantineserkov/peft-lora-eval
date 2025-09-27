from transformers import AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict
from src.logger import get_logger

logger = get_logger()


def configure_peft_model(model_name: str, config_dict: Dict, device: str) -> torch.nn.Module:
    logger.debug("configuring a model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
        offload_folder="offload"
    )

    if config_dict['method'] == "base":
        logger.info("PEFT is not used.")
    elif config_dict['method'] == "lora":
        logger.info("PEFT method: LoRA")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config_dict['r'],
            lora_alpha=config_dict['peft_config']['lora_alpha'],
            use_rslora=config_dict['peft_config']['use_rslora'],
            target_modules=config_dict['peft_config']['target_modules']
        )
        model = get_peft_model(model, peft_config)
    elif config_dict['method'] == "qlora":
        logger.info("PEFT method: QLoRA")
        pass
    elif config_dict['method'] == "dora":
        logger.info("PEFT method: DoRA")
        pass
    elif config_dict['method'] == "qdora":
        logger.info("PEFT method: QDoRA")
        pass
    else:
        logger.error("There is no way that you see this error message")
        raise ValueError

    return model.to(device)