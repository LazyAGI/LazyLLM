import os
import base64

import lazyllm
from lazyllm import launchers, LazyLLMCMD, bind, _0
from ..base import LazyLLMDeployBase
from ..lightllm import get_url_form_job

import cloudpickle


class RelayServer(LazyLLMDeployBase):
    def __init__(self, pre_func= None, post_func=None, port=17784, *, launcher=launchers.slurm):
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

        cmd = f'python {run_file_path} --target_url={url} --open_port={self.port} '
        if self.pre:
            cmd += f'--before_function="{self.pre}" '
        if self.post:
            cmd += f'--after_function="{self.post}" '
        
        return LazyLLMCMD(cmd=cmd,
                          post_function=bind(get_url_form_job, _0, self.port),
                          no_displays=['before_function','after_function']
                         )
    