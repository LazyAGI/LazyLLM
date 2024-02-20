import os
import random
import base64

import lazyllm
from lazyllm import launchers, LazyLLMCMD, bind, _0
from ..base import LazyLLMDeployBase
from ..lightllm import restart_service

import cloudpickle


class RelayServer(LazyLLMDeployBase):
    def __init__(self, pre_func= None, post_func=None, port=None, *, launcher=launchers.slurm):
        if pre_func:
            serialized_function = cloudpickle.dumps(pre_func)
            self.pre = base64.b64encode(serialized_function).decode('utf-8')
        else:
            self.pre = pre_func
        if post_func:
            serialized_function = cloudpickle.dumps(post_func)
            self.post = base64.b64encode(serialized_function).decode('utf-8')
        else:
            self.post = post_func
        self.port = port
        super().__init__(launcher=launcher)

    def cmd(self, url):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        run_file_path = os.path.join(folder_path, 'server.py')

        def build_cmd():
            port = self.port if self.port else random.randint(30000, 40000)
            cmd = f'python {run_file_path} --target_url={url} --open_port={port} '
            if self.pre:
                cmd += f'--before_function="{self.pre}" '
            if self.post:
                cmd += f'--after_function="{self.post}" '
            return cmd, port

        cmd, port = build_cmd()
        func = build_cmd if not self.port else None
        return LazyLLMCMD(cmd=cmd,
                          post_function=bind(restart_service, _0, port, func),
                          no_displays=['before_function','after_function']
                         )
    