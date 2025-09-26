import logging
import colorlog

# configure a single global handler
_handler = logging.StreamHandler()
_handler.setFormatter(colorlog.ColoredFormatter(
"%(log_color)s%(levelname)-8s%(reset)s %(name)s: %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
))

# root logger config
logging.basicConfig(level=logging.DEBUG, handlers=[_handler])

def get_logger(name: str = __name__):
    return logging.getLogger(name)
