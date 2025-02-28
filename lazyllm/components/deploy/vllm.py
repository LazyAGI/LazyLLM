import os
import sys
import json
import random
import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func
from .utils import get_log_path, make_log_dir


class Vllm(LazyLLMDeployBase):
    keys_name_handle = {
        'inputs': 'prompt',
        'stop': 'stop'
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'stop': ['<|im_end|>', '<|im_start|>', '</s>', '<|assistant|>', '<|user|>', '<|system|>', '<eos>'],
        'skip_special_tokens': False,
        'temperature': 0.6,
        'top_p': 0.8,
        'max_tokens': 4096
    }
    auto_map = {'tp': 'tensor-parallel-size'}

    def __init__(self, trust_remote_code=True, launcher=launchers.remote(ngpus=1), stream=False, log_path=None, **kw):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'dtype': 'auto',
            'kv-cache-dtype': 'auto',
            'tokenizer-mode': 'auto',
            'device': 'auto',
            'block-size': 16,
            'tensor-parallel-size': 1,
            'seed': 0,
            'port': 'auto',
            'host': '0.0.0.0',
            'max-num-seqs': 256,
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'vllm') if log_path else None

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

            cmd = f'{sys.executable} -m vllm.entrypoints.api_server --model {finetuned_model} '
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust-remote-code '
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
        return json.loads(x)['text'][0]

    @staticmethod
    def stream_parse_parameters():
        return {"decode_unicode": False, "delimiter": b"\0"}

    @staticmethod
    def stream_url_suffix():
        return ''
