import os
import json
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func
from .utils import make_log_dir, get_log_path


class Lightllm(LazyLLMDeployBase):
    keys_name_handle = {
        'inputs': 'inputs',
        'stop': 'stop_sequences'
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'inputs': 'Who are you ?',
        'parameters': {
            'do_sample': False,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            'temperature': 1.0,
            "top_p": 1,
            "top_k": -1,  # -1 is for all
            "ignore_eos": False,
            'max_new_tokens': 8192,
            "stop_sequences": None,
        }
    }
    auto_map = {'tp': 'tp'}

    def __init__(self, trust_remote_code=True, launcher=launchers.remote(ngpus=1), log_path=None, **kw):
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
            "data_type": 'float16',
            "max_req_total_len": 64000,
            "max_req_input_len": 4096,
            "long_truncation_mode": "head",
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] else True
        self.random_nccl_port = False if 'nccl_port' in kw and kw['nccl_port'] else True
        self.temp_folder = make_log_dir(log_path, 'lightllm') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)
            if self.random_nccl_port:
                self.kw['nccl_port'] = random.randint(20000, 30000)
            cmd = f'python -m lightllm.server.api_server --model_dir {finetuned_model} '
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust_remote_code '
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
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
    def extract_result(x, inputs):
        try:
            if x.startswith("data:"): return json.loads(x[len("data:"):])['token']['text']
            else: return json.loads(x)['generated_text'][0]
        except Exception as e:
            LOG.warning(f'JSONDecodeError on load {x}')
            raise e

    @staticmethod
    def stream_parse_parameters():
        return {"delimiter": b"\n\n"}

    @staticmethod
    def stream_url_suffix():
        return "_stream"
