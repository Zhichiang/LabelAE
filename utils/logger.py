# -----------------------------------------------------------------------------
# Create by Zhichiang
# -----------------------------------------------------------------------------

import sys
import logging


class LoggerSetup:
    file_name = ""


def setup_logger(name, save_to_dir=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_to_dir:
        fh = logging.FileHandler(LoggerSetup.file_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
