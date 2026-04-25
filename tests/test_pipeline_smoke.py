from unittest.mock import MagicMock, call, patch

import src.runner.pipeline as pipeline_module
from src.runner.common import resolve_stages


def test_run_pipeline_reuses_cached_model_for_same_method(tmp_path):
    logger = MagicMock()
    metadata = {
        "device": "cpu",
        "completed": [],
        "metadata_path": str(tmp_path / "metadata.json"),
        "run_config": {"methods": ["base"], "stages": ["eval", "bench"]},
    }

    run_dir = tmp_path / "runs" / "run_smoke"
    run_dir.mkdir(parents=True)

    config = {
        "model": {"model_name_or_path": "dummy-model"},
        "runtime": {"merge": False},
    }

    first_base_model = MagicMock(name="first_base_model")
    recycled_base_model = MagicMock(name="recycled_base_model")
    eval_stage_model = MagicMock(name="eval_stage_model")
    bench_stage_model = MagicMock(name="bench_stage_model")

    with (
        patch.object(pipeline_module, "init_run", return_value=(run_dir, metadata, logger)),
        patch.object(
            pipeline_module,
            "resolve_stages",
            return_value=([("base", "eval"), ("base", "bench")], []),
        ),
        patch.object(pipeline_module, "load_method_config", return_value=config) as load_method_config,
        patch.object(pipeline_module, "init_base_model", return_value=first_base_model) as init_base_model,
        patch.object(pipeline_module, "run_eval", return_value=eval_stage_model) as run_eval,
        patch.object(pipeline_module, "run_bench", return_value=bench_stage_model) as run_bench,
        patch.object(
            pipeline_module,
            "cleanup_model_for_cache",
            side_effect=[recycled_base_model, recycled_base_model],
        ) as cleanup_model_for_cache,
        patch.object(pipeline_module, "save_metadata") as save_metadata,
        patch.object(pipeline_module.gc, "collect"),
        patch.object(pipeline_module.torch.cuda, "is_available", return_value=False),
    ):
        pipeline_module.run_pipeline()

    load_method_config.assert_called_once_with("base", metadata["run_config"], logger)
    init_base_model.assert_called_once_with("base", config, logger)
    run_eval.assert_called_once_with(first_base_model, "base", config, metadata, logger)
    run_bench.assert_called_once_with(recycled_base_model, "base", config, metadata, logger)
    cleanup_model_for_cache.assert_has_calls(
        [
            call(eval_stage_model, config, logger),
            call(bench_stage_model, config, logger),
        ]
    )
    assert metadata["completed"] == ["base_eval", "base_bench"]
    assert save_metadata.call_count == 2


def test_run_pipeline_bypasses_cache_for_merged_eval_stages(tmp_path):
    logger = MagicMock()
    metadata = {
        "device": "cpu",
        "completed": [],
        "metadata_path": str(tmp_path / "metadata.json"),
        "run_config": {"methods": ["lora"], "stages": ["eval", "bench"]},
    }

    run_dir = tmp_path / "runs" / "run_merge_smoke"
    run_dir.mkdir(parents=True)

    config = {
        "model": {"model_name_or_path": "dummy-model"},
        "runtime": {"merge": True},
    }

    eval_base_model = MagicMock(name="eval_base_model")
    bench_base_model = MagicMock(name="bench_base_model")
    eval_stage_model = MagicMock(name="eval_stage_model")
    bench_stage_model = MagicMock(name="bench_stage_model")

    with (
        patch.object(pipeline_module, "init_run", return_value=(run_dir, metadata, logger)),
        patch.object(
            pipeline_module,
            "resolve_stages",
            return_value=([("lora", "eval"), ("lora", "bench")], []),
        ),
        patch.object(pipeline_module, "load_method_config", return_value=config) as load_method_config,
        patch.object(
            pipeline_module,
            "init_base_model",
            side_effect=[eval_base_model, bench_base_model],
        ) as init_base_model,
        patch.object(pipeline_module, "run_eval", return_value=eval_stage_model) as run_eval,
        patch.object(pipeline_module, "run_bench", return_value=bench_stage_model) as run_bench,
        patch.object(pipeline_module, "cleanup_model_for_cache") as cleanup_model_for_cache,
        patch.object(pipeline_module, "save_metadata") as save_metadata,
        patch.object(pipeline_module.gc, "collect"),
        patch.object(pipeline_module.torch.cuda, "is_available", return_value=False),
    ):
        pipeline_module.run_pipeline()

    load_method_config.assert_called_once_with("lora", metadata["run_config"], logger)
    init_base_model.assert_has_calls(
        [
            call("lora", config, logger),
            call("lora", config, logger),
        ]
    )
    run_eval.assert_called_once_with(eval_base_model, "lora", config, metadata, logger)
    run_bench.assert_called_once_with(bench_base_model, "lora", config, metadata, logger)
    cleanup_model_for_cache.assert_not_called()
    assert metadata["completed"] == ["lora_eval", "lora_bench"]
    assert save_metadata.call_count == 2


def test_resolve_stages_returns_skipped_completed_tags():
    metadata = {
        "completed": ["dora_train", "dora_eval"],
        "run_config": {
            "methods": ["dora", "qdora"],
            "stages": ["train", "eval"],
        },
    }

    plan, skipped = resolve_stages(metadata)

    assert plan == [("qdora", "train"), ("qdora", "eval")]
    assert skipped == ["dora_train", "dora_eval"]
