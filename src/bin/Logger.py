"""
This python file contains code to handle and define the logging of the tool.
"""

# ----------------------------
# ---------- IMPORT ----------
# ----------------------------
import logging as log


# ----------------------------
# ------ FORMAT LOGGING ------
# ----------------------------
class CustomFormatter(log.Formatter):
    """
    Class to set custom colors and format to the logger.
    """
    grey = "\x1b[1;37m"
    blue = "\x1b[34m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        log.DEBUG: grey + format + reset,
        log.INFO: blue + format + reset,
        log.WARNING: yellow + format + reset,
        log.ERROR: red + format + reset,
        log.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = log.Formatter(log_fmt)
        return formatter.format(record)


# ----------------------------
# ------ CUSTOM LOGGER -------
# ----------------------------
class CustomLogger:
    """
    Class to initiate the logger.
    """
    def __init__(self):
        logfile_path = "logfile.txt"
        log.basicConfig(level=log.DEBUG, filename=logfile_path, format="%(asctime)s - %(message)s", filemode="w")
        stderr_logger = log.StreamHandler()
        stderr_logger.setLevel(log.DEBUG)
        stderr_logger.setFormatter(CustomFormatter())
        log.getLogger().addHandler(stderr_logger)
