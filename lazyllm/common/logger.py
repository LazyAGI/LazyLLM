import inspect
import logging
from json import JSONDecodeError, loads
from os import getenv, getpid, listdir
import os
from os.path import join
from sys import stderr
from typing import Dict
from zipfile import ZipFile
import lazyllm
import platform
from .utils import check_path
from .common import call_once, once_flag

from loguru import logger

lazyllm.config.add("debug", bool, False, "DEBUG")
lazyllm.config.add("log_name", str, "lazyllm", "LOG_NAME")
lazyllm.config.add("log_level", str, "INFO", "LOG_LEVEL")
lazyllm.config.add(
    "log_format",
    str,
    "{process}: <green>{time:YYYY-MM-DD HH:mm:ss}</green> {extra[name]} "
    "<level>{level}</level>: ({name}:{line}) <cyan>{message}</cyan>",
    "LOG_FORMAT",
)
lazyllm.config.add("log_dir", str, os.path.join(os.path.expanduser('~'), '.lazyllm'), "LOG_DIR")
lazyllm.config.add("log_file_level", str, "ERROR", "LOG_FILE_LEVEL")
lazyllm.config.add("log_file_size", str, "4 MB", "LOG_FILE_SIZE")
lazyllm.config.add("log_file_retention", str, "7 days", "LOG_FILE_RETENTION")
lazyllm.config.add("log_file_mode", str, "merge", "LOG_FILE_MODE")


class _Log:
    _stderr_initialized = False
    _once_flags: Dict = {}

    def __init__(self):
        self._name = lazyllm.config["log_name"]
        self._pid = getpid()
        self._log_dir_path = check_path(
            lazyllm.config["log_dir"], exist=False, file=False
        )

        if getenv("LOGURU_AUTOINIT", "true").lower() in ("1", "true") and stderr:
            try:
                logger.remove(0)
            except ValueError:
                pass

        if not _Log._stderr_initialized:
            # A sink that will accumulate the log and output to stderr.
            self.stderr: bool = bool(stderr)
            self._stderr_i = logger.add(
                stderr,
                level=(
                    lazyllm.config["log_level"]
                    if not lazyllm.config["debug"]
                    else "DEBUG"
                ),
                format=lazyllm.config["log_format"],
                filter=lambda record: (
                    record["extra"].get("name") == self._name and self.stderr
                ),
                colorize=True,
            )
            _Log._stderr_initialized = True

        self._logger = logger.bind(name=self._name, process=self._pid)

    def log_once(self, message: str, level: str = "warning") -> None:
        frame = inspect.currentframe().f_back
        context = (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
        if context not in self._once_flags:
            self._once_flags[context] = once_flag()
        # opt depth for printing correct stack depth information
        call_once(
            self._once_flags[context],
            getattr(self.opt(depth=2, record=True).bind(name=self._name), level),
            message,
        )

    def read(self, limit: int = 10, level: str = "error"):
        names = listdir(self._log_dir_path)
        lines = []
        for name in names:
            if name.endswith(".json.log"):
                with open(join(self._log_dir_path, name)) as file:
                    lines = file.readlines()
            elif name.endswith(".json.log.zip"):
                with ZipFile(name) as zip_file:
                    for n in zip_file.namelist():
                        with zip_file.open(n, "r") as file:
                            lines = file.readlines()
        records = []
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        for line in lines:
            try:
                record = loads(line)
                if record:
                    record = record["record"]
                    no = record["level"]["no"]
                    if no >= level:
                        records.append(record)
            except JSONDecodeError:
                pass
        records = sorted(records, key=lambda r: r["time"]["timestamp"])
        if limit > 0:
            records = records[-limit:]
        return records

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self._logger, attr)
        return getattr(self, attr)

    def close(self):
        logger.remove()

    def __reduce__(self):
        return (self.__class__, ())


LOG = _Log()


def add_file_sink():
    name = lazyllm.config["log_name"]
    pid = getpid()
    log_dir_path = LOG._log_dir_path
    if log_dir_path:
        log_file_mode = lazyllm.config["log_file_mode"]
        if log_file_mode == "merge":
            log_file_name = f"{name}.json.log"
            enqueue = True
        elif log_file_mode == "split":
            log_file_name = f"{name}.{pid}.json.log"
            enqueue = False
        else:
            raise ValueError(f"Unexpected log_file_mode: {log_file_mode}")

        log_file_path = join(log_dir_path, log_file_name)
        LOG.add(
            log_file_path,
            level=lazyllm.config["log_file_level"],
            format="{message}",
            encoding="utf-8",
            rotation=lazyllm.config["log_file_size"],
            retention=lazyllm.config["log_file_retention"],
            compression="zip",
            delay=True,
            enqueue=enqueue,  # multiprocessing-safe
            colorize=True,
            serialize=True,
            filter=lambda record: (record["extra"].get("name") == name),
        )


add_file_sink()

if platform.system() != "Windows":
    os.register_at_fork(
        after_in_child=add_file_sink,
    )
