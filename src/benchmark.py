from typing import Dict, Any
import lm_eval
from lm_eval.models.huggingface import HFLM
from src.results import save_results


DEFAULT_BENCHMARK_METRICS = {
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "mmlu": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
}

def run_lm_eval(model, tokenizer, config_dict: Dict) -> Dict[str, Any]:
    batch_size = config_dict["benchmark"]["batch_size"]

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    return lm_eval.evaluator.simple_evaluate(
        model=lm,
        tasks=config_dict["benchmark"]["tasks"],
        batch_size=batch_size,
        no_cache=True,
    )


def extract_key_metrics(
    raw_results: Dict[str, Dict[str, Any]],
    metric_mapping: Dict[str, str],
    logger,
) -> Dict[str, float]:
    summary = {}

    for task_name, metric_key in metric_mapping.items():
        task_result = raw_results.get(task_name)

        if task_result is None:
            logger.warning(
                "Benchmark task '%s' is missing from lm-eval results. Available tasks: %s",
                task_name,
                sorted(raw_results.keys()),
            )
            continue

        if metric_key not in task_result:
            logger.warning(
                "Benchmark metric '%s' is missing for task '%s'. Available keys: %s",
                metric_key,
                task_name,
                sorted(task_result.keys()),
            )
            continue

        summary[task_name] = round(float(task_result[metric_key]) * 100, 2)

    return summary

def run_benchmarks(model, tokenizer, config, device, logger, metadata):
    metadata["metric_type"] = "benchmark"

    lm_eval_output = run_lm_eval(model, tokenizer, config)
    raw_results = lm_eval_output["results"]

    metric_mapping = config["benchmark"].get(
        "metrics",
        DEFAULT_BENCHMARK_METRICS,
    )

    summary = extract_key_metrics(raw_results, metric_mapping, logger)

    metrics = {
        "summary": summary,
        "raw": raw_results,
        "configs": lm_eval_output.get("configs"),
        "versions": lm_eval_output.get("versions"),
        "n_shot": lm_eval_output.get("n-shot"),
    }

    save_results(metrics, metadata, config, logger)
    return metrics