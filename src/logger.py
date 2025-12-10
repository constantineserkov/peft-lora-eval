import logging
import colorlog
import os
import pynvml
import atexit
import sys
from pathlib import Path

from accelerate.commands.menu.selection_menu import in_colab


# configure a single global handler
def set_up_logging(
        log_filename: str | Path = "runs/default_log_dir/project.log"
):
    from src.utils import in_colab, in_kaggle
    local = True
    if in_colab() or in_kaggle():
        logging.debug("Working in colab")
        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        local = False

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(name)s: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
    ))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    if local:
        # root logger config
        logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])

    # Root logger: clear any old handlers first
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # NVML init here (once per call, but since module-level, once per import)
    try:
        pynvml.nvmlInit()
        logging.debug("NVML initialized in logger")  # Or use your log.info
        # Auto-shutdown on process exit (decrements refcount)
        atexit.register(pynvml.nvmlShutdown)
    except Exception as e:
        logging.warning(f"Failed to initialize NVML in logger: {e}. VRAM logging disabled.")


def get_logger(name: str = __name__, level: str = "INFO"):
    # setLevel debug is temporary
    logger = logging.getLogger(name)

    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    logger.setLevel(numeric_level)

    return logger

