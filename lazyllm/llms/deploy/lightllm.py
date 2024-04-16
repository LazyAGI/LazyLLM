import os
import random

from lazyllm import launchers, LazyLLMCMD, bind, _0, ArgsDict
from .base import LazyLLMDeployBase, verify_fastapi_func


class Lightllm(LazyLLMDeployBase):
    input_key_name = 'inputs'
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        input_key_name: 'Who are you ?',
        'parameters': {
            'do_sample': False,
            "presence_penalty":0.0, 
            "frequency_penalty":0.0, 
            "repetition_penalty":1.0, 
            'temperature': 1.0,
            "top_p":1, 
            "top_k":-1, # -1 is for all
            "ignore_eos": False, 
            'max_new_tokens': 512,
            "stop_sequences":None
        }
    }

    def __init__(self,
                 trust_remote_code=True,
                 launcher=launchers.slurm,
                 stream=False,
                 **kw,
                 ):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'tp': 1,
            'max_total_token_num': 64000,
            'eos_id': 2,
            'port': None,
            'host':'0.0.0.0',
            'nccl_port': None,
            'tokenizer_mode': 'auto',
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)
        

    def cmd(self, model_dir=None, base_model=None):
        if not os.path.exists(model_dir) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(model_dir)):
            if not model_dir:
                print(f"Note! That model_dir({model_dir}) is an invalid path, "
                    f"base_model({base_model}) will be used")
            model_dir = base_model

        def impl():
            if not self.kw['port']:
                self.kw['port'] = random.randint(30000, 40000)
            if not self.kw['nccl_port']:
                self.kw['nccl_port'] = random.randint(20000, 30000)
            cmd = f'python -m lightllm.server.api_server --model_dir {model_dir} '
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust_remote_code '
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'