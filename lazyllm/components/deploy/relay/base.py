import os
import random
import inspect
import sys
import shlex
import subprocess
import tempfile

from lazyllm import launchers, LazyLLMCMD, dump_obj, config
from ..base import LazyLLMDeployBase, verify_fastapi_func, verify_ray_func
from ..utils import get_log_path, make_log_dir
from typing import Optional

config.add('use_ray', bool, False, 'USE_RAY', description='Whether to use Ray for ServerModule(relay server).')


def _should_write_relay_arg_file():
    return os.name == 'nt'


def _can_use_relay_arg_files(launcher):
    return _should_write_relay_arg_file() and isinstance(launcher, launchers.EmptyLauncher)


def _quote_relay_cmd_arg(value):
    if os.name == 'nt':
        quoted = subprocess.list2cmdline([value])
        if not (quoted.startswith('"') and quoted.endswith('"')):
            quoted = f'"{quoted}"'
        return quoted.replace('%', '%%')
    return shlex.quote(value)


def _relay_payload_arg(name, value, *, use_file=None):
    if not value:
        return ''
    if use_file is None:
        use_file = _should_write_relay_arg_file()
    if use_file:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            prefix=f'lazyllm-relay-{name}-',
            suffix='.b64',
            delete=False,
        ) as handle:
            handle.write(value)
            file_path = handle.name
        return f'--{name}_file={_quote_relay_cmd_arg(file_path)} '
    return f'--{name}={_quote_relay_cmd_arg(value)} '


class RelayServer(LazyLLMDeployBase):
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}
    message_format = None

    def __init__(self, port=None, *, func=None, pre_func=None, post_func=None, pythonpath=None,
                 log_path=None, cls=None, launcher=launchers.remote(sync=False), num_replicas: int = 1,  # noqa B008
                 security_key: Optional[str] = None, defined_pos: Optional[str] = None):
        # func must dump in __call__ to wait for dependancies.
        self._func = func
        self._pre = dump_obj(pre_func)
        self._post = dump_obj(post_func)
        self._port, self._real_port = port, None
        self._pythonpath = pythonpath
        self._num_replicas = num_replicas
        self._security_key = security_key
        self._defined_pos = defined_pos
        super().__init__(launcher=launcher)
        self.temp_folder = make_log_dir(log_path, cls or 'relay') if log_path else None

    def cmd(self, func=None):
        FastapiApp.update()
        self._func = dump_obj(func or self._func)
        folder_path = os.path.dirname(os.path.abspath(__file__))
        run_file_path = os.path.join(folder_path, 'server.py')

        def impl():
            self._real_port = self._port if self._port else random.randint(30000, 40000)
            use_arg_file = _can_use_relay_arg_files(self._launcher)
            cmd = f'{_quote_relay_cmd_arg(sys.executable)} {_quote_relay_cmd_arg(run_file_path)} '
            cmd += f'--open_port={self._real_port} '
            cmd += _relay_payload_arg('function', self._func, use_file=use_arg_file)
            if self._pre:
                cmd += _relay_payload_arg('before_function', self._pre, use_file=use_arg_file)
            if self._post:
                cmd += _relay_payload_arg('after_function', self._post, use_file=use_arg_file)
            if self._pythonpath:
                cmd += _relay_payload_arg('pythonpath', self._pythonpath, use_file=use_arg_file)
            if self._num_replicas > 1 and config['use_ray']:
                cmd += f'--num_replicas={self._num_replicas} '
            if self._security_key:
                cmd += _relay_payload_arg('security_key', self._security_key, use_file=use_arg_file)
            if self._defined_pos:
                cmd += _relay_payload_arg('defined_pos', dump_obj(self._defined_pos.replace('"', r'\"')),
                                          use_file=use_arg_file)
            if self.temp_folder: cmd += f' 2>&1 | tee {_quote_relay_cmd_arg(get_log_path(self.temp_folder))}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl,
                          checkf=verify_ray_func if config['use_ray'] else verify_fastapi_func,
                          no_displays=['function', 'function_file',
                                       'before_function', 'before_function_file',
                                       'after_function', 'after_function_file',
                                       'defined_pos', 'defined_pos_file',
                                       'pythonpath_file',
                                       'security_key', 'security_key_file'])

    def geturl(self, job=None):
        if job is None:
            job = self.job
        return f'http://{job.get_jobip()}:{self._real_port}/generate'


class FastapiApp(object):
    __relay_services__ = []

    @staticmethod
    def _server(method, path, **kw):
        def impl(f):
            FastapiApp.__relay_services__.append([f, method, path, kw])
            return f
        return impl

    @staticmethod
    def get(path, **kw):
        return FastapiApp._server('get', path, **kw)

    @staticmethod
    def post(path, **kw):
        return FastapiApp._server('post', path, **kw)

    @staticmethod
    def list(path, **kw):
        return FastapiApp._server('list', path, **kw)

    @staticmethod
    def delete(path, **kw):
        return FastapiApp._server('delete', path, **kw)

    @staticmethod
    def update():
        for f, method, path, kw in FastapiApp.__relay_services__:
            cls = inspect._findclass(f)
            if '__relay_services__' not in cls.__dict__:
                cls.__relay_services__ = dict()
            cls.__relay_services__[method, path] = ([f.__name__, kw])
        FastapiApp.__relay_services__.clear()
