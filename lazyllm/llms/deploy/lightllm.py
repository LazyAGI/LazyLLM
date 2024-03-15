import time
import json
import random
import requests
import httpx

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

def evaluate(url):
    def impl(input):
        with httpx.Client(timeout=90) as client:
           response = client.post(url, json=input, headers={'Content-Type': 'application/json'})
        return response
    return impl

def verify_launch_server(job):
    while True:
        line = job.queue.get()
        if line.startswith('ERROR:'):
            print("Capture error message: ", line, "\n\n")
            return False, line
        elif 'Uvicorn running on' in line:
            print("Capture startup message:   ",line)
            break
        if job.status == lazyllm.launchers.status.Failed:
            return False, 'Service Startup Failed.'
    return True, line

def restart_service(job, port, build_cmd=None, count=0):
    assert count < 5, 'The service failed to restart 5 times.'
    
    if lazyllm.mode == lazyllm.Mode.Display:
        return f'http://{job.name}:{port}/generate'
    status = launchers.status
    with timeout(3600, msg='Launch failed: No computing resources are available.'):
        while job.status in (status.TBSubmitted, status.InQueue, status.Pending):
            time.sleep(2)
    
    assert job.status == lazyllm.launchers.status.Running, 'Job failed'
    url = f'http://{job.get_jobip()}:{port}/generate'

    with timeout(240, msg='Service encountered an unknown exception.'):
        verify_res, verify_str = verify_launch_server(job)
        if verify_res:
            _ = requests.post(url, headers=headers, data=json.dumps(data))
            job.return_value = url
            return url
        elif build_cmd:
            cmd, port = build_cmd()
            job.cmd = LazyLLMCMD(cmd=cmd)
            job.restart()
            return restart_service(job, port, build_cmd, count+1)
        else:
            raise verify_str
    

@register('deploy', cmd=True)
def lllmserver(model_dir=None, tp=1, max_total_token_num=64000, eos_id=2,
            port=None, host='0.0.0.0', nccl_port=None, tokenizer_mode='auto',
            trust_remote_code=True):

        def build_cmd():
            open_p = port if port else random.randint(30000, 40000)
            nccl_p = nccl_port if nccl_port else random.randint(20000, 30000)

            cmd = (
                'python -m lightllm.server.api_server '
                f'--model_dir {model_dir} '
                f'--tp {tp} '
                f'--nccl_port {nccl_p} '
                f'--max_total_token_num {max_total_token_num} '
                f'--tokenizer_mode "{tokenizer_mode}" '
                f'--port {open_p} '
                f'--host "{host}" '
                f'--eos_id {eos_id} '
            )
            if trust_remote_code:
                cmd += '--trust_remote_code '
            return cmd, open_p
        func = build_cmd if not port else None
        cmd, port = build_cmd()
        return LazyLLMCMD(cmd=cmd, post_function=bind(restart_service, _0, port, func))

class Lightllm(LazyLLMDeployBase, flows.NamedPipeline):
    def __init__(self,
                 model_dir=None,
                 tp=1,
                 max_total_token_num=64000,
                 eos_id=2,
                 pre_func=None,
                 post_func=None,
                 open_port=None,
                 llm_launcher=launchers.slurm,
                 relay_launcher=launchers.slurm):
        super().__init__(launcher=llm_launcher)
        self.model_dir = model_dir
        self.tp = tp
        self.max_total_token_num = max_total_token_num
        self.eos_id = eos_id
        self.pre_func = pre_func
        self.post_func = post_func
        self.open_port = open_port

        flows.NamedPipeline.__init__(self,
            deploy_stage1 = show_io,
            deploy_stage2 = bind(deploy.lllmserver(launcher=llm_launcher),
                                 _0,
                                 self.tp,
                                 self.max_total_token_num,
                                 self.eos_id
                                 ),
            deploy_stage3 = evaluate,
            deploy_stage4 = deploy.RelayServer(
                                pre_func=self.pre_func,
                                post_func=self.post_func,
                                port=self.open_port,
                                launcher=relay_launcher),
            deploy_stage5 = show_io,
	    )

    def __call__(self, base_model):
        url = flows.NamedPipeline.__call__(self, base_model)
        return url

    def __repr__(self):
        return flows.NamedPipeline.__repr__(self)
