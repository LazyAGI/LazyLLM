import os
import json
import random
import importlib.util

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG, config
from .base import LazyLLMDeployBase, verify_fastapi_func
from .utils import get_log_path, make_log_dir, parse_options_keys


config.add('lmdeploy_eager_mode', bool, False, 'LMDEPLOY_EAGER_MODE',
           description='Whether to use eager mode for lmdeploy.')

class LMDeploy(LazyLLMDeployBase):
    keys_name_handle = {
        'inputs': 'prompt',
        'stop': 'stop',
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'stop': None,
        'top_p': 0.8,
        'temperature': 0.8,
        'skip_special_tokens': True,
    }
    auto_map = {
        'port': 'server-port',
        'host': 'server-name',
        'max_batch_size': 'max-batch-size',
    }
    stream_parse_parameters = {'delimiter': b'\n'}

    def __init__(self, launcher=launchers.remote(ngpus=1), trust_remote_code=True, log_path=None, **kw):  # noqa B008
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'server-name': '0.0.0.0',
            'server-port': None,
            'tp': 1,
            'max-batch-size': 128,
        })
        self.options_keys = kw.pop('options_keys', [])
        self.kw.check_and_update(kw)
        self._trust_remote_code = trust_remote_code
        self.random_port = False if 'server-port' in kw and kw['server-port'] else True
        self.temp_folder = make_log_dir(log_path, 'lmdeploy') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                            f'base_model({base_model}) will be used')
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['server-port'] = random.randint(30000, 40000)
            cmd = f'lmdeploy serve api_server {finetuned_model} --model-name lazyllm '

            if importlib.util.find_spec('torch_npu') is not None: cmd += '--device ascend '
            if config['lmdeploy_eager_mode']: cmd += '--eager-mode '
            cmd += self.kw.parse_kwargs()
            cmd += ' ' + parse_options_keys(self.options_keys)
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/v1/'
        else:
            return f'http://{job.get_jobip()}:{self.kw["server-port"]}/v1/'

    @staticmethod
    def extract_result(x, inputs):
        return json.loads(x)['text']
