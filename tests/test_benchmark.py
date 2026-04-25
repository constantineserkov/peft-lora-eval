from pathlib import Path

import pytest

import src.benchmark as benchmark


def _benchmark_config(tmp_path):
    return {
        "runtime": {"base_path": str(tmp_path)},
        "model": {"model_name_or_path": "org/model"},
        "active_method": "qdora",
        "benchmark": {
            "batch_size": 1,
            "tasks": ["arc_challenge"],
            "num_fewshot": None,
            "limit": 1,
        },
    }


def test_resolve_lm_eval_cache_path_converts_true_to_run_cache_path(tmp_path):
    config = _benchmark_config(tmp_path)

    cache_path = benchmark._resolve_lm_eval_cache_path(
        {"use_cache": True},
        config,
        {"run_id": "run_test_1"},
    )

    assert cache_path == str(
        tmp_path / "runs" / "run_test_1" / "qdora" / "lm_eval_cache" / "org_model"
    )
    assert Path(cache_path).parent.exists()


@pytest.mark.parametrize("use_cache", [False, None, ""])
def test_resolve_lm_eval_cache_path_disables_cache(use_cache, tmp_path):
    assert (
        benchmark._resolve_lm_eval_cache_path(
            {"use_cache": use_cache},
            _benchmark_config(tmp_path),
        )
        is None
    )


def test_resolve_lm_eval_cache_path_rejects_unsupported_value(tmp_path):
    with pytest.raises(ValueError, match="benchmark.use_cache"):
        benchmark._resolve_lm_eval_cache_path(
            {"use_cache": 1},
            _benchmark_config(tmp_path),
        )


def test_run_lm_eval_passes_cache_path_not_boolean(monkeypatch, tmp_path):
    config = _benchmark_config(tmp_path)
    config["benchmark"]["use_cache"] = True

    captured = {}

    monkeypatch.setattr(
        benchmark,
        "HFLM",
        lambda pretrained, tokenizer, batch_size: "wrapped-lm",
    )

    def fake_simple_evaluate(**kwargs):
        captured.update(kwargs)
        return {"results": {}}

    monkeypatch.setattr(benchmark.evaluator, "simple_evaluate", fake_simple_evaluate)

    benchmark.run_lm_eval(
        model=object(),
        tokenizer=object(),
        config_dict=config,
        device="cpu",
        metadata={"run_id": "run_test_1"},
    )

    assert captured["model"] == "wrapped-lm"
    assert captured["use_cache"] == str(
        tmp_path / "runs" / "run_test_1" / "qdora" / "lm_eval_cache" / "org_model"
    )
    assert not isinstance(captured["use_cache"], bool)
