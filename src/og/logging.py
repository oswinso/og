import sys
from inspect import getframeinfo, stack
from typing import Callable

from loguru import logger

FORMAT = (
    "<green>{time:MM-DD HH:mm:ss.S}</green> | "
    "<level>{level.icon}</level> | "
    "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
COMPILE_FORMAT = "<green>{time:MM-DD HH:mm:ss.S}</green> | <level>{message}</level>"

log = logger


def set_logger_format():
    log.remove()
    log.add(sys.stdout, format=FORMAT, level="TRACE", filter=lambda record: "compile_log" not in record["extra"])
    log.add(sys.stdout, format=COMPILE_FORMAT, level="TRACE", filter=lambda record: "compile_log" in record["extra"])

    levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    for level_name in levels:
        log.level(level_name, icon=level_name[0])


_log_one_set = set()


def log_once(log_fn: Callable, message: str, *args, **kwargs):
    caller = getframeinfo(stack()[1][0])
    caller_id = "{}:{}".format(caller.filename, caller.lineno)
    if caller_id in _log_one_set:
        return
    log_fn(message, *args, **kwargs)
    _log_one_set.add(caller_id)
