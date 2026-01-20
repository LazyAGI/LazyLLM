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
from ..utils import check_path
from ..common import call_once, once_flag

from loguru import logger

lazyllm.config.add('debug', bool, False, 'DEBUG', description='Whether to enable debug mode.')
lazyllm.config.add('log_name', str, 'lazyllm', 'LOG_NAME', description='The name of the log file.')
lazyllm.config.add('expected_log_modules', str, 'lazyllm', 'EXPECTED_LOG_MODULES',
                   description='The expected log modules, separated by comma.')
lazyllm.config.add('log_level', str, 'INFO', 'LOG_LEVEL', description='The level of the log.')
lazyllm.config.add('log_format', str, 'long', 'LOG_FORMAT', description='The format of the log.')
lazyllm.config.add('log_dir', str, os.path.join(os.path.expanduser(lazyllm.config['home']), 'logs'), 'LOG_DIR',
                   description='The directory of the log file.')
lazyllm.config.add('log_file_level', str, 'ERROR', 'LOG_FILE_LEVEL', description='The level of the log file.')
lazyllm.config.add('log_file_size', str, '4 MB', 'LOG_FILE_SIZE', description='The size of the log file.')
lazyllm.config.add('log_file_retention', str, '7 days', 'LOG_FILE_RETENTION',
                   description='The retention of the log file.')
lazyllm.config.add('log_file_mode', str, 'merge', 'LOG_FILE_MODE', description='The mode of the log file.')


def _get_log_format(fmt: str):
    if not fmt or fmt in [None, 'default', 'long']:
        return ('<green>{time:YYYY-MM-DD HH:mm:ss}</> {extra[name]} <level>{level}</> '
                '({name}:{line}, {process}{extra[jobid]}): <cyan>{message}</>')
    elif fmt == 'short': return '<green>{time:YYYY-MM-DD HH:mm:ss}</> <level>{level}</>: <cyan>{message}</>'
    else: return fmt


def _get_expected_log_modules():
    if (log_name := lazyllm.config['expected_log_modules']) is None:
        return None
    if log_name == 'all': return None
    return [n.strip() for n in log_name.split(',')]

class _Log:
    _stderr_initialized = False
    _once_flags: Dict = {}
    __dynamic_attrs__ = ['debug', 'info', 'warning', 'error', 'success', 'critical']

    def __init__(self):
        self._pid = getpid()
        self._expected = _get_expected_log_modules()
        self._log_dir_path = check_path(lazyllm.config['log_dir'], exist=False, file=False)

        if getenv('LOGURU_AUTOINIT', 'true').lower() in ('1', 'true') and stderr:
            try:
                logger.remove(0)
            except ValueError:
                pass

        if not _Log._stderr_initialized:
            # A sink that will accumulate the log and output to stderr.
            self.stderr: bool = bool(stderr)
            self._stderr_i = logger.add(
                stderr,
                level=lazyllm.config['log_level'] if not lazyllm.config['debug'] else 'DEBUG',
                format=_get_log_format(lazyllm.config['log_format']), colorize=True,
                filter=lambda rd: (not self._expected or rd['extra'].get('name') in self._expected) and self.stderr)
            _Log._stderr_initialized = True

        self._logger = logger.bind(name='lazyllm', jobid='')

    def log_once(self, message: str, level: str = 'warning', **kw) -> None:
        frame = inspect.currentframe().f_back
        context = (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
        if context not in self._once_flags:
            self._once_flags[context] = once_flag()

        jobid = f', jobid={kw["jobid"]}' if 'jobid' in kw else ''
        # opt depth for printing correct stack depth information
        message = message.replace('{', '{{').replace('}', '}}')
        call_once(self._once_flags[context],
                  getattr(self.opt(depth=2, record=True).bind(name=(kw.get('name') or 'lazyllm'),
                                                              jobid=jobid), level), message)

    def read(self, limit: int = 10, level: str = 'error'):
        names = listdir(self._log_dir_path)
        lines = []
        for name in names:
            if name.endswith('.json.log'):
                with open(join(self._log_dir_path, name)) as file:
                    lines = file.readlines()
            elif name.endswith('.json.log.zip'):
                with ZipFile(name) as zip_file:
                    for n in zip_file.namelist():
                        with zip_file.open(n, 'r') as file:
                            lines = file.readlines()
        records = []
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        for line in lines:
            try:
                record = loads(line)
                if record:
                    record = record['record']
                    no = record['level']['no']
                    if no >= level:
                        records.append(record)
            except JSONDecodeError:
                pass
        records = sorted(records, key=lambda r: r['time']['timestamp'])
        if limit > 0:
            records = records[-limit:]
        return records

    def __getattr__(self, attr):
        def impl(*args, join: str = '\n', depth: int = 0, **kw):
            call_kw = dict(jobid=(f', jobid={jobid}' if (jobid := kw.pop('jobid', None)) else ''))
            if 'name' in kw: call_kw['name'] = kw.pop('name')
            s = str(args[0]) if len(args) == 1 else join.join([str(a) for a in args])
            s = s.replace('{', '{{').replace('}', '}}')
            getattr(self._logger.opt(depth=depth + 1, **kw), attr)(s, **call_kw)

        return impl if attr in self.__dynamic_attrs__ else getattr(self._logger, attr)

    def close(self):
        logger.remove()

    def __reduce__(self):
        return (self.__class__, ())

    def __dir__(self):
        return super().__dir__() + self.__dynamic_attrs__

LOG = _Log()


def add_file_sink():
    name = lazyllm.config['log_name']
    excepted = _get_expected_log_modules()
    pid = getpid()
    log_dir_path = LOG._log_dir_path
    if log_dir_path:
        log_file_mode = lazyllm.config['log_file_mode']
        if log_file_mode == 'merge':
            log_file_name = f'{name}.json.log'
            enqueue = True
        elif log_file_mode == 'split':
            log_file_name = f'{name}.{pid}.json.log'
            enqueue = False
        else:
            raise ValueError(f'Unexpected log_file_mode: {log_file_mode}')

        log_file_path = join(log_dir_path, log_file_name)
        LOG.add(
            log_file_path,
            level=lazyllm.config['log_file_level'],
            format='{message}',
            encoding='utf-8',
            rotation=lazyllm.config['log_file_size'],
            retention=lazyllm.config['log_file_retention'],
            compression='zip',
            delay=True,
            enqueue=enqueue,  # multiprocessing-safe
            colorize=True,
            serialize=True,
            filter=(lambda rd: not excepted or rd['extra'].get('name') in excepted)
        )


add_file_sink()

if platform.system() != 'Windows':
    os.register_at_fork(
        after_in_child=add_file_sink,
    )
