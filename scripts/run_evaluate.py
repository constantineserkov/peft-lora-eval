import time
import argparse
from src.logger import set_up_logging, get_logger


# Wall-clock time start
wc_start = time.time()

metadata = {
    "wc_start": wc_start,

}


def main():
    # setup logging
    set_up_logging()

    # load config