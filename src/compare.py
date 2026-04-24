import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
import pandas as pd


DISPLAY_EXCLUDED_COLUMNS = {"source_path"}


def get_results_root(config: Dict[str, Any]) -> Path:
    return Path(config["runtime"]["base_path"]) / "results"


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_result_files(results_root: Path, metric_type: str) -> Iterable[Path]:
    if metric_type == "evaluator":
        result_dir = results_root / "metrics"
    elif metric_type == "benchmark":
        result_dir = results_root / "benchmarks"
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    if not result_dir.exists():
        return []

    return result_dir.glob("*.json")


def load_results_for_run(
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    metric_type: str,
) -> List[Dict[str, Any]]:
    results_root = get_results_root(config)
    run_id = metadata["run_id"]

    rows = []
    for path in iter_result_files(results_root, metric_type):
        result = load_json_file(path)

        result_metadata = result.get("metadata", {})
        if result_metadata.get("run_id") != run_id:
            continue

        if result_metadata.get("metric_type") != metric_type:
            continue

        result["_source_path"] = str(path)
        rows.append(result)

    return rows


def build_eval_row(result: Dict[str, Any]) -> Dict[str, Any]:
    config = result["config"]
    metrics = result["metrics"]

    evaluation = metrics.get("evaluation", {})
    hardware = metrics.get("hardware", {})
    time_metrics = metrics.get("time", {})

    return {
        "method": config.get("active_method"),
        "model": config.get("model", {}).get("model_name_or_path"),
        "avg_loss": evaluation.get("avg_loss"),
        "perplexity": evaluation.get("perplexity"),
        "min_loss": evaluation.get("min_loss"),
        "max_loss": evaluation.get("max_loss"),
        "peak_vram_torch_gb": hardware.get("peak_vram_torch_gb"),
        "avg_vram_gb": hardware.get("avg_vram"),
        "elapsed_seconds": time_metrics.get("test_elapsed_time"),
        "timestamp": result.get("timestamp"),
        "source_path": result.get("_source_path"),
    }


def build_eval_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [build_eval_row(result) for result in results]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.sort_values(["method", "timestamp"])


def build_benchmark_row(result: Dict[str, Any]) -> Dict[str, Any]:
    config = result["config"]
    metrics = result["metrics"]

    summary = metrics.get("summary", {})
    benchmark = metrics.get("benchmark", {})

    row = {
        "method": config.get("active_method"),
        "model": config.get("model", {}).get("model_name_or_path"),
        "elapsed_seconds": benchmark.get("elapsed_seconds"),
        "peak_vram_torch_gb": benchmark.get("peak_vram_torch_gb"),
        "batch_size": benchmark.get("batch_size"),
        "timestamp": result.get("timestamp"),
        "source_path": result.get("_source_path"),
    }

    for task_name, score in summary.items():
        row[task_name] = score

    return row


def build_benchmark_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [build_benchmark_row(result) for result in results]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.sort_values(["method", "timestamp"])


def save_comparison_tables(
    eval_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    results_root = get_results_root(config)
    comparison_dir = results_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    run_id = metadata["run_id"]

    paths = {}

    if not eval_df.empty:
        eval_path = comparison_dir / f"{run_id}_evaluation.csv"
        eval_df.to_csv(eval_path, index=False)
        paths["evaluation"] = str(eval_path)

    if not benchmark_df.empty:
        benchmark_path = comparison_dir / f"{run_id}_benchmarks.csv"
        benchmark_df.to_csv(benchmark_path, index=False)
        paths["benchmark"] = str(benchmark_path)

    summary_path = comparison_dir / f"{run_id}_summary.json"
    paths["summary"] = str(summary_path)

    summary = {
        "run_id": run_id,
        "evaluation_rows": len(eval_df),
        "benchmark_rows": len(benchmark_df),
        "paths": paths,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return paths


def format_comparison_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"

    display_df = df.drop(
        columns=[column for column in DISPLAY_EXCLUDED_COLUMNS if column in df.columns],
        errors="ignore",
    )

    return display_df.to_string(index=False)


def print_comparison_tables(
    eval_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    logger,
) -> None:
    logger.info("Evaluation comparison table:\n%s", format_comparison_table(eval_df))
    logger.info("Benchmark comparison table:\n%s", format_comparison_table(benchmark_df))


def create_comparison_tables(
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    eval_results = load_results_for_run(config, metadata, "evaluator")
    benchmark_results = load_results_for_run(config, metadata, "benchmark")

    eval_df = build_eval_table(eval_results)
    benchmark_df = build_benchmark_table(benchmark_results)

    paths = save_comparison_tables(eval_df, benchmark_df, config, metadata)
    print_comparison_tables(eval_df, benchmark_df, logger)

    logger.info(
        "Comparison tables created. Evaluation rows: %s | Benchmark rows: %s",
        len(eval_df),
        len(benchmark_df),
    )

    return {
        "evaluation_rows": len(eval_df),
        "benchmark_rows": len(benchmark_df),
        "paths": paths,
    }

