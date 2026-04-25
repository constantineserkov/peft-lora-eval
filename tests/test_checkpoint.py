import pickle
from pathlib import Path
from unittest.mock import MagicMock

import src.checkpoint as checkpoint_module


def _checkpoint_context(tmp_path):
    checkpoint_path = (
        tmp_path
        / "runs"
        / "run_resume"
        / "dora"
        / "checkpoints"
        / "resume"
        / "checkpoint.pt"
    )
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")

    config = {
        "active_method": "dora",
        "runtime": {"base_path": str(tmp_path)},
    }
    metadata = {
        "run_id": "run_resume",
        "latest_checkpoint": "checkpoint.pt",
        "latest_checkpoint_method": "dora",
    }

    return config, metadata, checkpoint_path


def test_load_checkpoint_uses_weights_only_safe_load(tmp_path, monkeypatch):
    config, metadata, checkpoint_path = _checkpoint_context(tmp_path)
    logger = MagicMock()
    loaded_checkpoint = {"epoch": 1, "step": 2}

    def fake_load(path, **kwargs):
        assert Path(path) == checkpoint_path
        assert kwargs == {"weights_only": True}
        return loaded_checkpoint

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    assert checkpoint_module.load_checkpoint(config, metadata, logger) == loaded_checkpoint
    logger.warning.assert_not_called()


def test_load_checkpoint_falls_back_for_legacy_pickled_config(tmp_path, monkeypatch):
    config, metadata, checkpoint_path = _checkpoint_context(tmp_path)
    logger = MagicMock()
    loaded_checkpoint = {"epoch": 1, "step": 2}
    calls = []

    def fake_load(path, **kwargs):
        assert Path(path) == checkpoint_path
        calls.append(kwargs)

        if kwargs == {"weights_only": True}:
            raise pickle.UnpicklingError("Weights only load failed")

        assert kwargs == {"weights_only": False}
        return loaded_checkpoint

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    assert checkpoint_module.load_checkpoint(config, metadata, logger) == loaded_checkpoint
    assert calls == [{"weights_only": True}, {"weights_only": False}]
    logger.warning.assert_called_once()
