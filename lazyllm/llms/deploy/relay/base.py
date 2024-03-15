import os
import random
import base64

import lazyllm
from lazyllm import launchers, LazyLLMCMD, bind, _0
from ..base import LazyLLMDeployBase
from ..lightllm import restart_service

import cloudpickle

def dump_func(f, old_value=None):
    f = old_value if f is None else f
    return None if f is None else base64.b64encode(cloudpickle.dumps(f)).decode('utf-8')


class RelayServer(LazyLLMDeployBase):
    def __init__(self, port=None, *, func=None, pre_func=None, post_func=None,
                 launcher=launchers.slurm(sync=False)):
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

        def build_cmd():
            port = self.port if self.port else random.randint(30000, 40000)
            cmd = f'python {run_file_path} --open_port={port} '
            cmd += f'--function="{self.func}" '
            if self.pre:
                cmd += f'--before_function="{self.pre}" '
            if self.post:
                cmd += f'--after_function="{self.post}" '
            return cmd, port

        cmd, port = build_cmd()
        func = build_cmd if not self.port else None
        return LazyLLMCMD(cmd=cmd,
                          post_function=bind(restart_service, _0, port, func),
                          no_displays=['function', 'before_function', 'after_function']
                         )
    