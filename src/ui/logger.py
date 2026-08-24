"""
HSR frontend custom logger
"""

import logging
import os
import sys


def get_custom_logger(logger_name: str) -> logging.Logger:
    message_format = "%(name)s:%(levelname)s: %(message)s"
    ui_logger = create_logger(name=logger_name, format_string=message_format)

    return ui_logger


def create_logger(
    name: str,
    format_string: str = "%(asctime)s:%(levelname)s:%(module)s.py:%(lineno)d:%(message)s",
) -> logging.Logger:
    level = os.environ.get("LOG_LEVEL", "INFO")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True

    # If this logger has been setup already, skip adding handlers
    if logger.handlers:
        for hdl in logger.handlers:
            hdl.setLevel(level)
    else:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        # add formatter to ch
        ch.setFormatter(logging.Formatter(fmt=format_string))
        logger.addHandler(ch)

    return logger
