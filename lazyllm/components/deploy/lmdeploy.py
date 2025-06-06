import os
import json
import random
import importlib.util

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func
from ..utils import ModelManager
from .utils import get_log_path, make_log_dir


class LMDeploy(LazyLLMDeployBase):
    keys_name_handle = {
        'inputs': 'prompt',
        'stop': 'stop',
        'image': 'image_url',
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        "image_url": None,
        "session_id": -1,
        "interactive_mode": False,
        "stream": False,
        "stop": None,
        "request_output_len": None,
        "top_p": 0.8,
        "top_k": 40,
        "temperature": 0.8,
        "repetition_penalty": 1,
        "ignore_eos": False,
        "skip_special_tokens": True,
        "cancel": False,
        "adapter_name": None
    }
    auto_map = {}

    def __init__(self, launcher=launchers.remote(ngpus=1), log_path=None, **kw):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'server-name': '0.0.0.0',
            'server-port': None,
            'tp': 1,
            "max-batch-size": 128,
            "chat-template": None,
        })
        self.kw.check_and_update(kw)
        self.random_port = False if 'server-port' in kw and kw['server-port'] else True
        self.temp_folder = make_log_dir(log_path, 'lmdeploy') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model

        model_type = ModelManager.get_model_type(base_model or finetuned_model)
        if model_type == 'vlm':
            self.kw.pop("chat-template")
        else:
            if not self.kw["chat-template"] and 'vl' not in finetuned_model and 'lava' not in finetuned_model:
                self.kw["chat-template"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'lmdeploy', 'chat_template.json')
            else:
                self.kw.pop("chat-template")

        def impl():
            if self.random_port:
                self.kw['server-port'] = random.randint(30000, 40000)
            cmd = f"lmdeploy serve api_server {finetuned_model} "

            if importlib.util.find_spec("torch_npu") is not None:
                cmd += "--device ascend --eager-mode "

            cmd += self.kw.parse_kwargs()
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/v1/chat/interactive'
        else:
            return f'http://{job.get_jobip()}:{self.kw["server-port"]}/v1/chat/interactive'

    @staticmethod
    def extract_result(x, inputs):
        return json.loads(x)['text']

    @staticmethod
    def stream_parse_parameters():
        return {"delimiter": b"\n"}

    @staticmethod
    def stream_url_suffix():
        return ''
