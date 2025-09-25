from transformers import AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict


def configure_peft_model(model_name: str, config_dict: Dict, device: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
        offload_folder="offload"
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config_dict['r'],
        lora_alpha=config_dict['lora_alpha'],
        use_rslora=config_dict['use_rslora'],
        target_modules=config_dict['target_modules']
    )

    model = get_peft_model(model, peft_config)
    return model.to(device)