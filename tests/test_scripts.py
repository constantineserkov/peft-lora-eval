import subprocess
import sys
from pathlib import Path


def test_run_train_with_temp_config(tmp_path):
    # Create a dummy config at tmp_path
    config_file = tmp_path / "dummy_config.yaml"
    config_file.write_text("mode: test\nmethod: base\nseed: 17")  # Minimal YAML
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_train", "--method", "base", "--device", "cpu"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )

    assert "config_path:" in result.stdout or result.stderr
    assert result.returncode == 0
    assert "Seed is set to" in result.stdout or result.stderr


def test_invalid_method_config(tmp_path):
    """Ensure invalid method raises ValueError."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_train", "--method", "INVALID_METHOD"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    # expect non-zero return
    assert result.returncode != 0
    assert "Invalid method" in result.stderr


def test_run_train_lora():
    """Ensure run_train.py runs end-to-end in test mode with LoRA method."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_train", "--method", "lora", "--mode", "test", "--data-subset", "10"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )

    assert result.returncode == 0
    assert "config_path:" in result.stdout or result.stderr
    assert "Seed is set to" in result.stdout or result.stderr
    assert "Device:" in result.stdout or result.stderr  # Basic log check
    assert "Successfully parsed args" in result.stdout or result.stderr  # Arg parsing check


def test_run_train_qlora():
    """Ensure run_train.py runs end-to-end in test mode with QLoRA method."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_train", "--method", "qlora", "--mode", "test", "--data-subset", "10"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )

    assert result.returncode == 0
    assert "config_path:" in result.stdout or result.stderr
    assert "Seed is set to" in result.stdout or result.stderr
    assert "Device:" in result.stdout or result.stderr
    assert "Successfully parsed args" in result.stdout or result.stderr


def test_run_train_qdora():
    """Ensure run_train.py runs end-to-end in test mode with QDoRA method."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_train", "--method", "qdora", "--mode", "test", "--data-subset", "10"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )

    assert result.returncode == 0
    assert "config_path:" in result.stdout or result.stderr
    assert "Seed is set to" in result.stdout or result.stderr
    assert "Device:" in result.stdout or result.stderr
    assert "Successfully parsed args" in result.stdout or result.stderr