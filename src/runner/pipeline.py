from transformers.models.auto.image_processing_auto import model_type

from src.model_utils import init_base_model, cleanup_model_for_cache
from src.utils import load_method_config, get_model_cache_key
from src.runner.common import init_run, resolve_stages, save_metadata
from src.runner.train_runner import run_train
from src.runner.eval_runner import run_eval
from src.runner.bench_runner import run_bench
from src.runner.inf_runner import run_inf


def run_pipeline():
    run_dir, metadata, logger = init_run()
    plan = resolve_stages(metadata)

    run_config = metadata["run_config"]

    # model = None
    # cur_method = None
    # config = None
    model_cache = {}
    config_cache = {}

    run = {
        "train": run_train,
        "eval": run_eval,
        "bench": run_bench,
        "inf": run_inf,
    }

    for method, stage in plan:
        # if method != cur_method:
        #     config = load_method_config(method, run_config, logger)
        #     model = init_base_model(method, config, logger)
        #     cur_method = method

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
            if model_key not in model_cache:
                model_cache[model_key] = init_base_model(method, config, logger)

            model = model_cache[model_key]
        else:
            logger.warning("merge=True disables model cache for this stage. Initializing the base model.")
            model = init_base_model(method, config, logger)

        stage_model = run[stage](model, method, config, metadata, logger)

        if stage_model is None:
            # stage_model = model
            raise RuntimeError(f"Stage '{stage}' did not return a model")

        if use_cache:
            clean_model = cleanup_model_for_cache(stage_model, config, logger)

            if clean_model is None:
                model_cache.pop(model_key, None)
            else:
                model_cache[model_key] = clean_model

        metadata["completed"].append(f"{method}_{stage}")
        save_metadata(metadata)

    # Comparison runner