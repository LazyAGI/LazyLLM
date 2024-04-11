import os
import random
import base64
from lazyllm import launchers, LazyLLMCMD
from ..base import LazyLLMDeployBase, verify_fastapi_func
from lazyllm.thirdparty import cloudpickle


def dump_func(f, old_value=None):
    f = old_value if f is None else f
    return None if f is None else base64.b64encode(cloudpickle.dumps(f)).decode('utf-8')


class RelayServer(LazyLLMDeployBase):
    input_key_name = None
    default_headers = {'Content-Type': 'application/json'}
    message_format = None

    def __init__(self, port=None, *, func=None, pre_func=None, post_func=None,
                 launcher=launchers.remote(sync=False)):
        # func must dump in __call__ to wait for dependancies.
        self.func = func
        self.pre = dump_func(pre_func)
        self.post = dump_func(post_func)
        self.port = port
        super().__init__(launcher=launcher)

    def cmd(self, func=None):
        self.func = dump_func(func, self.func)
        folder_path = os.path.dirname(os.path.abspath(__file__))
        run_file_path = os.path.join(folder_path, 'server.py')

        def impl():
            self.port = self.port if self.port else random.randint(30000, 40000)
            cmd = f'python {run_file_path} --open_port={self.port} --function="{self.func}" '
            if self.pre:
                cmd += f'--before_function="{self.pre}" '
            if self.post:
                cmd += f'--after_function="{self.post}" '
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func,
                          no_displays=['function', 'before_function', 'after_function'])

    def geturl(self, job=None):
        if job is None:
            job = self.job
        return f'http://{job.get_jobip()}:{self.port}/generate'
