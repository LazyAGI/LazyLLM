import os
import random
import inspect
import shlex
import subprocess
import sys
import tempfile

from lazyllm import launchers, LazyLLMCMD, dump_obj, config
from ..base import LazyLLMDeployBase, verify_fastapi_func, verify_ray_func
from ..utils import get_log_path, make_log_dir
from typing import Optional
import base64

config.add('use_ray', bool, False, 'USE_RAY', description='Whether to use Ray for ServerModule(relay server).')
config.add('pass_args_by_file', bool, False, 'PASS_ARGS_BY_FILE',
           description='When True, serialised relay cmd args are always written to temp files.')

# Serialised objects longer than this (bytes) are written to a temp file so the
# OS command-line length limit is not exceeded.
_CMD_ARG_SIZE_THRESHOLD = 65536


class RelayServer(LazyLLMDeployBase):
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}
    message_format = None

    def __init__(self, port=None, *, func=None, pre_func=None, post_func=None, pythonpath=None,
                 log_path=None, cls=None, launcher=launchers.remote(sync=False), num_replicas: int = 1,  # noqa B008
                 security_key: Optional[str] = None, defined_pos: Optional[str] = None,
                 pass_args_by_file: Optional[bool] = None):
        # func must dump in __call__ to wait for dependancies.
        self._func = func
        self._pre = dump_obj(pre_func)
        self._post = dump_obj(post_func)
        self._port, self._real_port = port, None
        self._pythonpath = pythonpath
        self._num_replicas = num_replicas
        self._security_key = security_key
        self._defined_pos = defined_pos
        # None means fall back to global config at call time.
        self._pass_args_by_file = pass_args_by_file
        super().__init__(launcher=launcher)
        self.temp_folder = make_log_dir(log_path, cls or 'relay') if log_path else None

    @property
    def _file_args_forced(self) -> bool:
        if self._pass_args_by_file is not None:
            return self._pass_args_by_file
        return config['pass_args_by_file']

    def _prepare_obj_arg(self, serialised: Optional[str], *, force: bool = False) -> Optional[str]:
        '''Return *serialised* as-is when small, or spill it to a temp file and
        return a ``@file:<path>`` reference when it exceeds the threshold.
        Pass ``force=True`` to always write to a file regardless of size.'''
        if serialised is None:
            return None
        if not force and len(serialised) <= _CMD_ARG_SIZE_THRESHOLD:
            return serialised
        temp_dir = os.path.abspath(config['temp_dir'])
        os.makedirs(temp_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix='.pkl', prefix='lazyllm_relay_', dir=temp_dir)
        os.close(fd)
        raw = base64.b64decode(serialised.encode('utf-8'))
        with open(path, 'wb') as fp:
            fp.write(raw)
        return f'@file:{path}'

    @staticmethod
    def _join_command(args) -> str:
        if os.name == 'nt':
            return subprocess.list2cmdline([str(arg) for arg in args])
        return shlex.join([str(arg) for arg in args])

    def cmd(self, func=None):
        FastapiApp.update()
        self._func = dump_obj(func or self._func)
        folder_path = os.path.dirname(os.path.abspath(__file__))
        run_file_path = os.path.join(folder_path, 'server.py')

        def impl():
            self._real_port = self._port if self._port else random.randint(30000, 40000)
            force = self._file_args_forced
            func_arg = self._prepare_obj_arg(self._func, force=force)
            args = [
                sys.executable,
                run_file_path,
                f'--open_port={self._real_port}',
                f'--function={func_arg}',
            ]
            if self._pre:
                args.append(f'--before_function={self._prepare_obj_arg(self._pre, force=force)}')
            if self._post:
                args.append(f'--after_function={self._prepare_obj_arg(self._post, force=force)}')
            if self._pythonpath:
                args.append(f'--pythonpath={self._pythonpath}')
            if self._num_replicas > 1 and config['use_ray']:
                args.append(f'--num_replicas={self._num_replicas}')
            if self._security_key:
                args.append(f'--security_key={self._security_key}')
            if self._defined_pos:
                defined_pos = dump_obj(self._defined_pos.replace('"', r'\"'))
                args.append(f'--defined_pos={self._prepare_obj_arg(defined_pos, force=force)}')
            cmd = self._join_command(args)
            if self.temp_folder:
                log_path = get_log_path(self.temp_folder)
                if os.name == 'nt':
                    cmd += f' > {subprocess.list2cmdline([log_path])} 2>&1'
                else:
                    cmd += f' 2>&1 | tee {shlex.quote(log_path)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl,
                          checkf=verify_ray_func if config['use_ray'] else verify_fastapi_func,
                          no_displays=['function', 'before_function', 'after_function', 'security_key', 'defined_pos'])

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
