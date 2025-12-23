import os
import random
import inspect
import sys

from lazyllm import launchers, LazyLLMCMD, dump_obj, config
from ..base import LazyLLMDeployBase, verify_fastapi_func, verify_ray_func
from ..utils import get_log_path, make_log_dir
from typing import Optional

config.add('use_ray', bool, False, 'USE_RAY', description='Whether to use Ray for ServerModule(relay server).')


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
            cmd = f'{sys.executable} {run_file_path} --open_port={self._real_port} --function="{self._func}" '
            if self._pre:
                cmd += f'--before_function="{self._pre}" '
            if self._post:
                cmd += f'--after_function="{self._post}" '
            if self._pythonpath:
                cmd += f'--pythonpath="{self._pythonpath}" '
            if self._num_replicas > 1 and config['use_ray']:
                cmd += f'--num_replicas={self._num_replicas}'
            if self._security_key:
                cmd += f'--security_key="{self._security_key}" '
            if self._defined_pos:
                cmd += '--defined_pos="{}" '.format(dump_obj(self._defined_pos.replace('"', r'\"')))
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
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
