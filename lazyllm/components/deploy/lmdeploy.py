import os
import json
import random
import importlib.util

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG, config
from .base import LazyLLMDeployBase, verify_fastapi_func
from ..utils import ModelManager
from .utils import get_log_path, make_log_dir


config.add('lmdeploy_eager_mode', bool, False, 'LMDEPLOY_EAGER_MODE')

class LMDeploy(LazyLLMDeployBase):
    """    This class is a subclass of ``LazyLLMDeployBase``, leveraging the inference capabilities provided by the [LMDeploy](https://github.com/InternLM/lmdeploy) framework for inference on large language models.

Args:
    launcher (lazyllm.launcher): The launcher for fine-tuning, defaults to ``launchers.remote(ngpus=1)``.
    stream (bool): Whether to enable streaming response, defaults to ``False``.
    kw: Keyword arguments for updating default training parameters. Note that no additional keyword arguments beyond those listed below can be passed.

Keyword Args: 
    tp (int): Tensor parallelism parameter, defaults to ``1``.
    server_name (str): The IP address of the service, defaults to ``0.0.0.0``.
    server_port (int): The port number of the service, defaults to ``None``. In this case, LazyLLM will automatically generate a random port number.
    max_batch_size (int): Maximum batch size, defaults to ``128``.



Examples:
    >>> # Basic use:
    >>> from lazyllm import deploy
    >>> infer = deploy.LMDeploy()
    >>>
    >>> # MultiModal:
    >>> import lazyllm
    >>> from lazyllm import deploy, globals
    >>> from lazyllm.components.formatter import encode_query_with_filepaths
    >>> chat = lazyllm.TrainableModule('Mini-InternVL-Chat-2B-V1-5').deploy_method(deploy.LMDeploy)
    >>> chat.update_server()
    >>> inputs = encode_query_with_filepaths('What is it?', ['path/to/image'])
    >>> res = chat(inputs)
    """
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
    auto_map = {
        'port': 'server-port',
        'host': 'server-name',
        'max_batch_size': 'max-batch-size',
        'chat_template': 'chat-template',
    }
    stream_parse_parameters = {"delimiter": b"\n"}

    def __init__(self, launcher=launchers.remote(ngpus=1), trust_remote_code=True, log_path=None, **kw):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'server-name': '0.0.0.0',
            'server-port': None,
            'tp': 1,
            "max-batch-size": 128,
            "chat-template": None,
        })
        self.kw.check_and_update(kw)
        self._trust_remote_code = trust_remote_code
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

            if importlib.util.find_spec("torch_npu") is not None: cmd += '--device ascend '
            if config['lmdeploy_eager_mode']: cmd += '--eager-mode '
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
