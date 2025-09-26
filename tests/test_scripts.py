import pytest
import yaml
import subprocess
import sys
from pathlib import Path




def test_run_train_with_temp_config(temp_config):
    """Ensure run_train.py runs with a temporary config."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_train", "--method", "base"],
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