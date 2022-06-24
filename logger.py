import logging
import sys

from config import settings

log = logging.getLogger()
for handler in log.handlers[:]:
    log.removeHandler(handler)

levels = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}

print("logger initialization is started.")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(settings.log_setting.format,
                                                   datefmt=settings.log_setting.date_format))
    return console_handler


def get_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(levels[settings.log_setting.level])
    logger.addHandler(get_console_handler())
    logger.propagate = True
    return logger
