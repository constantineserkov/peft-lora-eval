import gc
import torch.cuda
from src.model_utils import init_base_model, cleanup_model_for_cache
from src.utils import load_method_config, get_model_cache_key
from src.runner.common import init_run, resolve_stages, save_metadata
from src.runner.train_runner import run_train
from src.runner.eval_runner import run_eval
from src.runner.bench_runner import run_bench
from src.runner.inf_runner import run_inf
from src.compare import create_comparison_tables

def run_pipeline():
    run_dir, metadata, logger = init_run()
    plan = resolve_stages(metadata)

    run_config = metadata["run_config"]

    cached_model = None
    cached_model_key = None
    config_cache = {}

    run = {
        "train": run_train,
        "eval": run_eval,
        "bench": run_bench,
        "inf": run_inf,
    }

    for method, stage in plan:
        if method not in config_cache:
            config_cache[method] = load_method_config(method, run_config, logger)

        config = config_cache[method]
        model_key = get_model_cache_key(config)

        use_cache = not(
            config["runtime"]["merge"]
            and stage in {"eval", "bench", "inf"}
            and method not in {"base", "instruct"}
        )

        if use_cache:
            if cached_model_key != model_key:
                if  cached_model is not None:
                    logger.info("Model config changed. Releasing cached model.")
                    del cached_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                cached_model = init_base_model(method, config, logger)
                cached_model_key = model_key

            model = cached_model
        else:
            logger.warning("merge=True disables model cache for this stage. Initializing the base model.")
            model = init_base_model(method, config, logger)

        stage_model = run[stage](model, method, config, metadata, logger)

        if stage_model is None:
            # stage_model = model
            raise RuntimeError(f"Stage '{stage}' did not return a model")

        if use_cache:
            clean_model = cleanup_model_for_cache(stage_model, config, logger)
            logger.info(f"Returned model to cache for key: {model_key}")

            if clean_model is None:
                logger.warning("Could not clean model for reuse. Releasing cached model.")
                del stage_model
                cached_model = None
                cached_model_key = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                cached_model = clean_model
        else:
            del stage_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metadata["completed"].append(f"{method}_{stage}")
        save_metadata(metadata)

    # Comparison runner
    create_comparison_tables(run_config, metadata, logger)
    logger.info("Pipeline ran successfully. Exiting.")