import logging
import colorlog
import os


# configure a single global handler
def set_up_logging(log_filename: str = "results/logs/project.log"):
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

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # root logger config
    logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])

def get_logger(name: str = __name__):
    return logging.getLogger(name)