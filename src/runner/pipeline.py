from src.model_utils import init_base_model
from src.utils import load_method_config
from src.runner.common import init_run, resolve_stages, save_metadata
from src.runner.train_runner import run_train
from src.runner.eval_runner import run_eval
from src.runner.bench_runner import run_bench
from src.runner.inf_runner import run_inf


def run_pipeline():
    run_dir, metadata, logger = init_run()
    plan = resolve_stages(metadata)

    run_config = metadata["run_config"]

    model = None
    cur_method = None

    run = {
        "train": run_train,
        "eval": run_eval,
        "bench": run_bench,
        "inf": run_inf,
    }

    for method, stage in plan:



        if method != cur_method:
            config = load_method_config(method, run_config, logger)
            model = init_base_model(method, config, logger)
            cur_method = method

        run[stage](model, method, config, metadata, logger)

        metadata["completed"].append(f"{method}_{stage}")
        save_metadata(metadata)

    # Comparison runner