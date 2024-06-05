import os
import json
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func


class Lightllm(LazyLLMDeployBase):
    input_key_name = 'inputs'
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        input_key_name: 'Who are you ?',
        'parameters': {
            'do_sample': False,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            'temperature': 1.0,
            "top_p": 1,
            "top_k": -1,  # -1 is for all
            "ignore_eos": False,
            'max_new_tokens': 512,
            "stop_sequences": None,
        }
    }
    auto_map = {'tp': 'tp'}

    def __init__(self,
                 trust_remote_code=True,
                 launcher=launchers.remote(ngpus=1),
                 stream=False,
                 **kw,
                 ):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'tp': 1,
            'max_total_token_num': 64000,
            'eos_id': 2,
            'port': None,
            'host': '0.0.0.0',
            'nccl_port': None,
            'tokenizer_mode': 'auto',
            "running_max_req_size": 256,
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)

    def cmd(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model

        def impl():
            if not self.kw['port']:
                self.kw['port'] = random.randint(30000, 40000)
            if not self.kw['nccl_port']:
                self.kw['nccl_port'] = random.randint(20000, 30000)
            cmd = f'python -m lightllm.server.api_server --model_dir {finetuned_model} '
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust_remote_code '
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/generate'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'

    @staticmethod
    def extract_result(x):
        return json.loads(x)['generated_text'][0]
