from pathlib import Path
from typing import Dict, Any
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from src.results import save_results
from src.metrics import measure_runtime_and_peak_vram
from src.utils import project_path


DEFAULT_BENCHMARK_METRICS = {
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "mmlu": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
    "gsm8k": "exact_match,strict-match"
}

def _resolve_lm_eval_cache_path(
    benchmark_config: Dict[str, Any],
    config_dict: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> str | None:
    use_cache = benchmark_config.get("use_cache")

    if use_cache in (None, False):
        return None

    if use_cache is True:
        method = config_dict.get("active_method", "benchmark")
        model_name = config_dict.get("model", {}).get("model_name_or_path", "model")
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")

        if metadata and metadata.get("run_id"):
            cache_path = project_path(
                config_dict,
                "runs",
                metadata["run_id"],
                method,
                "lm_eval_cache",
                safe_model_name,
            )
        else:
            cache_path = project_path(
                config_dict,
                "results",
                "lm_eval_cache",
                method,
                safe_model_name,
            )
    elif isinstance(use_cache, str):
        cache_path = use_cache.strip()
        if not cache_path:
            return None
    else:
        raise ValueError(
            "benchmark.use_cache must be false/null, true, or a cache path string."
        )

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def _resolve_lm_eval_seed(
    benchmark_config: Dict[str, Any],
    config_dict: Dict[str, Any],
) -> int | None:
    seed = benchmark_config.get("seed", config_dict["runtime"]["seed"])

    if seed is None:
        return None

    if not isinstance(seed, int):
        raise ValueError("benchmark.seed/runtime.seed must be an integer or null.")

    return seed


def run_lm_eval(
    model,
    tokenizer,
    config_dict: Dict,
    device,
    metadata: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:

    batch_size = config_dict["benchmark"]["batch_size"]

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    with measure_runtime_and_peak_vram(device) as benchmark_stats:
        benchmark_config = config_dict["benchmark"]
        seed = _resolve_lm_eval_seed(benchmark_config, config_dict)

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=benchmark_config["tasks"],
            num_fewshot=benchmark_config.get("num_fewshot"),
            batch_size=batch_size,
            use_cache=_resolve_lm_eval_cache_path(
                benchmark_config,
                config_dict,
                metadata,
            ),
            limit=benchmark_config.get("limit"),
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )

    benchmark_stats["batch_size"] = batch_size
    benchmark_stats["seed"] = seed

    return results, benchmark_stats


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

    lm_eval_output, benchmark_stats = run_lm_eval(
        model,
        tokenizer,
        config,
        device,
        metadata,
    )
    raw_results = lm_eval_output["results"]

    metric_mapping = config["benchmark"].get(
        "metrics",
        DEFAULT_BENCHMARK_METRICS,
    )

    summary = extract_key_metrics(raw_results, metric_mapping, logger)

    metrics = {
        "summary": summary,
        "benchmark": benchmark_stats,
        "raw": raw_results,
        "configs": lm_eval_output.get("configs"),
        "versions": lm_eval_output.get("versions"),
        "n_shot": lm_eval_output.get("n-shot"),
    }

    save_results(metrics, metadata, config, logger)
    return metrics
