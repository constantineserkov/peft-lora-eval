from pathlib import Path
from typing import List, Dict, Tuple
import json, datetime, time

from src.utils import (
    parse_args,
    load_and_validate_run_config,
    set_seed,
    resolve_device,
    resolve_base_path,
)
from src.logger import set_up_logging, get_logger
from src.auth import init_wandb, init_hf_auth


def init_run():
    # Local wall-clock time start that will add up (end-start) to wc_accumulated
    wc_attempt_start = datetime.datetime.now().isoformat()

    args = parse_args()
    base_path = resolve_base_path(args)
    args.base_path = base_path

    run_name = "run_" + args.run_id
    run_dir = Path(base_path) / "runs" / run_name

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["attempt"] += 1
        metadata["wc_attempt_start"] = wc_attempt_start
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": run_dir.name,
            "attempt": 1,
            "wc_attempt_start": wc_attempt_start,
            "wc_accumulated": {},  # accumulated per method_stage time
            "all_wc": [],  # accumulated times of all methods_stages in a list
            "method": None,  # lora/qlora/dora/qdora/base/instruct
            "stage": None,  # training/eval/bench/inf
            "completed": [],
            "latest_checkpoint": None,
            "last_global_step": 0,
            "note": None,
            "metadata_path": str(metadata_path),
            "immutable": {},  # Parameters that are fixed for a run_#. e.g. random seed
        }

    set_up_logging(run_dir / "run.log", level=args.log_level.upper())
    logger = get_logger(level=args.log_level.upper())

    device = resolve_device()
    logger.info(f"Device: {device}")

    run_config = load_and_validate_run_config(args, logger)
    set_seed(run_config["runtime"]["seed"], logger)

    init_wandb(run_config, logger)
    init_hf_auth(logger)

    metadata["device"] = device
    metadata["run_config"] = run_config

    metadata_path.write_text(json.dumps(metadata, indent=2))
    return run_dir, metadata, logger


def resolve_stages(metadata) -> List[Tuple[str, str]]:
    completed = set(metadata.get("completed", []))

    run_config = metadata["run_config"]
    methods: List = run_config["methods"]
    requested_stages = set(run_config["stages"])

    plan = []

    for method in methods:
        # Dependency always: eval/bench/inf require train
        need_train = "train" in requested_stages
        need_eval = "eval" in requested_stages
        need_bench = "bench" in requested_stages
        need_inf = "inf" in requested_stages

        # If eval or inf requested, enforce train unless base or instruct model -> (don't require training)
        if ((need_eval or need_inf or need_bench) and "train" not in requested_stages
                and method.lower() not in {"base", "instruct"}):
            need_train = True

        # BUILD PIPELINE ORDER
        ordered = []
        if need_train:
            ordered.append("train")
        if need_eval:
            ordered.append("eval")
        if need_bench:
            ordered.append("bench")
        if need_inf:
            ordered.append("inf")

        # CONVERT TO method_stage:
        for stage in ordered:
            tag = f"{method}_{stage}"
            if tag not in completed:
                plan.append((method, stage))

    return plan


def save_metadata(metadata):
    metadata_path = Path(metadata["metadata_path"])
    metadata["updated_time"] = time.time()
    metadata_path.write_text(json.dumps(metadata, indent=2))
