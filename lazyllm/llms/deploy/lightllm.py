import time
import json
import random
import requests
import os

import lazyllm
from lazyllm import launchers, flows, package, LazyLLMCMD, bind, _0, timeout
from .base import LazyLLMDeployBase
from ..core import register
try:
    from builtins import deploy
except Exception:
    from lazyllm import LazyLLMRegisterMetaClass
    deploy = LazyLLMRegisterMetaClass.all_groups['deploy']


headers = {'Content-Type': 'application/json'}
data = {
    'inputs': 'Who are you ?',
    "parameters": {
        'do_sample': False,
        'ignore_eos': False,
        'max_new_tokens': 512,
        'temperature': 0.1,
    }
}

def show_io(s21):
    print(f'input or output is: {s21}')
    return s21


def get_url_form_job(job, port):
    if lazyllm.mode == lazyllm.Mode.Display:
        return f'http://{job.name}:{port}/generate'
    status = launchers.status
    with timeout(3600, msg='Launch failed: No computing resources are available.'):
        while job.status in (status.TBSubmitted, status.InQueue, status.Pending):
            time.sleep(2)
    assert job.status == lazyllm.launchers.status.Running, 'Job failed'
    url = f'http://{job.get_jobip()}:{port}/generate'
    with timeout(600, msg='Service encountered an unknown exception.'):
        while True:
            try:
                _ = requests.post(url, headers=headers, data=json.dumps(data))
                return url
            except:
                time.sleep(5)
    

@register('deploy', cmd=True)
def lllmserver(model_dir=None, tp=1, max_total_token_num=64000, eos_id=2,
            port=None, host='0.0.0.0', nccl_port=None, tokenizer_mode='auto',
            trust_remote_code=True):
        port = port if port else random.randint(30000, 40000)
        nccl_port = nccl_port if nccl_port else random.randint(20000, 30000)

        cmd = (
            'python -m lightllm.server.api_server '
            f'--model_dir {model_dir} '
            f'--tp {tp} '
            f'--nccl_port {nccl_port} '
            f'--max_total_token_num {max_total_token_num} '
            f'--tokenizer_mode "{tokenizer_mode}" '
            f'--port {port} '
            f'--host "{host}" '
            f'--eos_id {eos_id} '
        )
        if trust_remote_code:
            cmd += '--trust_remote_code '
        return LazyLLMCMD(cmd=cmd, post_function=bind(get_url_form_job, _0, port))
    

class Lightllm(LazyLLMDeployBase, flows.NamedPipeline):
    def __init__(self,
                 model_dir=None,
                 tp=1,
                 max_total_token_num=64000,
                 eos_id=2,
                 launcher=launchers.slurm):
        super().__init__(launcher=launcher)
        self.model_dir = model_dir
        self.tp = tp
        self.max_total_token_num = max_total_token_num
        self.eos_id = eos_id

        flows.NamedPipeline.__init__(self,
            deploy_dd = flows.NamedPipeline(
                deploy_stage2 = show_io,
                deploy_stage3 = bind(deploy.lllmserver(launcher=launcher),
                                     _0,
                                     self.tp,
                                     self.max_total_token_num,
                                     self.eos_id
                                     ),
                deploy_stage4 = show_io,
            )
	)

    def __call__(self, base_model):
        url = flows.NamedPipeline.__call__(self, base_model)
        return url

    def __repr__(self):
        return flows.NamedPipeline.__repr__(self)
