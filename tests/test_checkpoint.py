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


def test_load_checkpoint_prefers_active_method_metadata(tmp_path, monkeypatch):
    config, metadata, _ = _checkpoint_context(tmp_path)
    dora_checkpoint_path = (
        tmp_path
        / "runs"
        / "run_resume"
        / "dora"
        / "checkpoints"
        / "resume"
        / "dora.pt"
    )
    dora_checkpoint_path.write_bytes(b"dora")

    metadata["latest_checkpoint"] = "qdora.pt"
    metadata["latest_checkpoint_method"] = "qdora"
    metadata["checkpoints"] = {
        "dora": {"latest_resume": "dora.pt"},
        "qdora": {"latest_resume": "qdora.pt"},
    }
    logger = MagicMock()
    loaded_checkpoint = {"epoch": 2, "step": 3}

    def fake_load(path, **kwargs):
        assert Path(path) == dora_checkpoint_path
        assert kwargs == {"weights_only": True}
        return loaded_checkpoint

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    assert checkpoint_module.load_checkpoint(config, metadata, logger) == loaded_checkpoint


def test_save_checkpoint_updates_per_method_metadata(tmp_path, monkeypatch):
    metadata = {
        "run_id": "run_resume",
        "metadata_path": str(tmp_path / "metadata.json"),
    }
    config = {
        "active_method": "dora",
        "runtime": {"base_path": str(tmp_path)},
        "peft_config": {"r": 8},
    }
    model = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    scaler = MagicMock()
    optimizer.state_dict.return_value = {"optimizer": "state"}
    scheduler.state_dict.return_value = {"scheduler": "state"}
    scaler.state_dict.return_value = {"scaler": "state"}

    monkeypatch.setattr(
        checkpoint_module,
        "get_peft_model_state_dict",
        lambda model: {"adapter": "state"},
    )
    monkeypatch.setattr(checkpoint_module.torch, "save", lambda checkpoint, path: None)
    save_metadata = MagicMock()
    monkeypatch.setattr(checkpoint_module, "save_metadata", save_metadata)

    checkpoint_module.save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=1,
        step=2,
        best_loss=0.5,
        config=config,
        metadata=metadata,
        logger=MagicMock(),
    )

    assert metadata["checkpoints"]["dora"]["latest_resume"] == "checkpoint_epoch_1_step_2.pt"
    assert metadata["checkpoints"]["dora"]["latest_resume_path"].endswith(
        "runs\\run_resume\\dora\\checkpoints\\resume\\checkpoint_epoch_1_step_2.pt"
    ) or metadata["checkpoints"]["dora"]["latest_resume_path"].endswith(
        "runs/run_resume/dora/checkpoints/resume/checkpoint_epoch_1_step_2.pt"
    )
    assert metadata["latest_checkpoint"] == "checkpoint_epoch_1_step_2.pt"
    assert metadata["latest_checkpoint_method"] == "dora"
    save_metadata.assert_called_once_with(metadata)


def test_save_best_updates_best_adapter_metadata(tmp_path, monkeypatch):
    metadata = {
        "run_id": "run_resume",
        "metadata_path": str(tmp_path / "metadata.json"),
    }
    config = {
        "active_method": "dora",
        "runtime": {"base_path": str(tmp_path)},
    }
    model = MagicMock()
    save_metadata = MagicMock()
    monkeypatch.setattr(checkpoint_module, "save_metadata", save_metadata)

    checkpoint_module.save_best(
        best_loss=0.5,
        model=model,
        epoch=1,
        step=2,
        config=config,
        metadata=metadata,
        logger=MagicMock(),
    )

    model.save_pretrained.assert_called_once_with(metadata["checkpoints"]["dora"]["best_adapter"])
    assert metadata["checkpoints"]["dora"]["best_adapter"].endswith(
        "runs\\run_resume\\dora\\checkpoints/best"
    ) or metadata["checkpoints"]["dora"]["best_adapter"].endswith(
        "runs/run_resume/dora/checkpoints/best"
    )
    save_metadata.assert_called_once_with(metadata)
