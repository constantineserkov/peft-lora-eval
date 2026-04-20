import os
from typing import Dict, Tuple
import torch
from transformers import AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from src.model_utils import configure_peft_model_for_eval
from src.logger import get_logger
from src.results import save_results

logger = get_logger()


# ------------------------------------------------------------------
# CONFIG — change only these
# ------------------------------------------------------------------
TASKS = [
    "arc_challenge",      # 25-shot
    "hellaswag",          # 10-shot
    "mmlu",               # 5-shot (57 subjects)
    "truthfulqa_mc2",     # 0-shot
    "winogrande",         # 5-shot
    "gsm8k",              # 8-shot
]

MODEL_PATHS = [
    "outputs/base-7B",
    "outputs/lora-r64",
    "outputs/qlora-4bit",
    "outputs/dora-r64",
    # add more...
]
# ------------------------------------------------------------------

def run_lm_eval(model, tokenizer, config_dict: Dict):
    batch_size = config_dict["dataloader"]["batch_size"]
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    results = lm_eval.evaluator.simple_evaluate(
        model=lm,
        tasks=config_dict["benchmark_tasks"],
        batch_size=batch_size,
        no_cache=True
    )
    return results["results"]


def extract_key_metrics(raw_results) -> Dict[str, float]:
    return {
        "MMLU": round(raw_results["mmlu"]["acc,none"] * 100, 2),
        "HellaSwag": round(raw_results["hellaswag"]["acc_norm,none"] * 100, 2),
        "ARC-Challenge": round(raw_results["arc_challenge"]["acc_norm,none"] * 100, 2),
        "TruthfulQA": round(raw_results["truthfulqa_mc2"]["acc,none"] * 100, 2),
        "Winogrande": round(raw_results["winogrande"]["acc,none"] * 100, 2),
        "GSM8K": round(raw_results["gsm8k"]["acc,none"] * 100, 2),
    }


def run_benchmarks(model, tokenizer, metadata, config, device):
    metadata["metric_type"] = "benchmark"

    all_results = {}
    os.makedirs("results/benchmarks", exist_ok=True)

    for idx, path in enumerate(MODEL_PATHS):
        try:
            model_name = os.path.basename(path)
            logger.info(f"\n{'=' * 60}\nBenchmarking {model_name}\n{'=' * 60}")
            logger.info(f"Model: {idx+1}/{len(MODEL_PATHS)}")

            raw_results = run_lm_eval(model, tokenizer, config)
            metrics = extract_key_metrics(raw_results)

            all_results[model_name] = metrics

            save_results(metrics, metadata, config)

            # clean GPU memory
            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Unexpected error: {e}.\nModels{len({MODEL_PATHS[idx:]})}: {MODEL_PATHS[idx:]} were not benchmarked")