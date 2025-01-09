import os
import random
import base64
import inspect
import sys

from lazyllm import launchers, LazyLLMCMD
from ..base import LazyLLMDeployBase, verify_fastapi_func
import cloudpickle
from contextlib import contextmanager
from ..utils import get_log_path, make_log_dir


def dump_func(f, old_value=None):
    @contextmanager
    def env_helper():
        os.environ['LAZYLLM_ON_CLOUDPICKLE'] = 'ON'
        yield
        os.environ['LAZYLLM_ON_CLOUDPICKLE'] = 'OFF'

    f = old_value if f is None else f
    with env_helper():
        return None if f is None else base64.b64encode(cloudpickle.dumps(f)).decode('utf-8')


class RelayServer(LazyLLMDeployBase):
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}
    message_format = None

    def __init__(self, port=None, *, func=None, pre_func=None, post_func=None,
                 pythonpath=None, log_path=None, cls=None, launcher=launchers.remote(sync=False)):
        # func must dump in __call__ to wait for dependancies.
        self.func = func
        self.pre = dump_func(pre_func)
        self.post = dump_func(post_func)
        self.port, self.real_port = port, None
        self.pythonpath = pythonpath
        super().__init__(launcher=launcher)
        self.temp_folder = make_log_dir(log_path, cls or 'relay') if log_path else None

    def cmd(self, func=None):
        FastapiApp.update()
        self.func = dump_func(func, self.func)
        folder_path = os.path.dirname(os.path.abspath(__file__))
        run_file_path = os.path.join(folder_path, 'server.py')

        def impl():
            self.real_port = self.port if self.port else random.randint(30000, 40000)
            cmd = f'{sys.executable} {run_file_path} --open_port={self.real_port} --function="{self.func}" '
            if self.pre:
                cmd += f'--before_function="{self.pre}" '
            if self.post:
                cmd += f'--after_function="{self.post}" '
            if self.pythonpath:
                cmd += f'--pythonpath="{self.pythonpath}" '
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func,
                          no_displays=['function', 'before_function', 'after_function'])

    def geturl(self, job=None):
        if job is None:
            job = self.job
        return f'http://{job.get_jobip()}:{self.real_port}/generate'


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
