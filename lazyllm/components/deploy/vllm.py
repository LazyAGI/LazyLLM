import os
import sys
import json
import random
import importlib.metadata
from packaging.version import parse
from typing import Optional

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG, LazyLLMLaunchersBase
from .base import LazyLLMDeployBase, verify_fastapi_func, verify_func_factory
from ...common import LazyLLMRegisterMetaClass
from .utils import get_log_path, make_log_dir
from .ray import reallocate_launcher, Distributed, sleep_moment


verify_vllm_openai_func = verify_func_factory(running_message='Application startup complete.')

class _VllmStreamParseParametersMeta(LazyLLMRegisterMetaClass):
    def __getattribute__(cls, name):
        if name == 'stream_parse_parameters':
            if not hasattr(cls, '_stream_parse_parameters'):
                try:
                    vllm_version = parse(importlib.metadata.version('vllm'))
                    cls._stream_parse_parameters = {'decode_unicode': False}
                    if vllm_version <= parse('0.5.0'): cls._stream_parse_parameters.update({'delimiter': b'\0'})
                except ImportError:
                    cls._stream_parse_parameters = {'decode_unicode': False}
            return cls._stream_parse_parameters
        return super().__getattribute__(name)


class Vllm(LazyLLMDeployBase, metaclass=_VllmStreamParseParametersMeta):
    """This class is a subclass of ``LazyLLMDeployBase``, leveraging the [VLLM](https://github.com/vllm-project/vllm) framework to deploy and run inference on large language models.

Args:
    trust_remote_code (bool): Whether to allow loading of model code from remote sources. Default is ``True``.
    launcher (lazyllm.launcher): The launcher used to start the model. Default is ``launchers.remote(ngpus=1)``.
    log_path (str): Path to store logs. If ``None``, logs will not be saved.
    openai_api (bool): Whether to start VLLM with OpenAI-compatible API. Default is ``False``.
    kw: Keyword arguments used to override default deployment parameters. No extra arguments beyond the supported ones are allowed.

The supported keyword arguments and their default values are as follows:

Keyword Args: 
    tensor-parallel-size (int): Tensor parallelism size. Default is ``1``.
    dtype (str): Data type for model weights and activations. Default is ``auto``. Options include: ``half``, ``float16``, ``bfloat16``, ``float``, ``float32``.
    kv-cache-dtype (str): Data type for KV cache. Default is ``auto``. Options include: ``fp8``, ``fp8_e5m2``, ``fp8_e4m3``.
    device (str): Backend device type supported by VLLM. Default is ``auto``. Options include: ``cuda``, ``neuron``, ``cpu``.
    block-size (int): Token block size. Default is ``16``.
    port (int | str): Service port number. Default is ``auto`` (random assignment).
    host (str): Service binding IP address. Default is ``0.0.0.0``.
    seed (int): Random seed. Default is ``0``.
    tokenizer_mode (str): Tokenizer loading mode. Default is ``auto``.
    max-num-seqs (int): Maximum number of concurrent requests supported by the inference engine. Default is ``256``.
    pipeline-parallel-size (int): Pipeline parallelism size. Default is ``1``.
    max-num-batched-tokens (int): Maximum number of batched tokens. Default is ``64000``.


Examples:
    >>> from lazyllm import deploy
    >>> infer = deploy.vllm()
    """
    # keys_name_handle/default_headers/message_format will lose efficacy when openai_api is True
    keys_name_handle = {'inputs': 'prompt', 'stop': 'stop'}
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
    optional_keys = set(['max-model-len'])

    # TODO(wangzhihong): change default value for `openai_api` argument to True
    def __init__(self, trust_remote_code: bool = True,
                 launcher: LazyLLMLaunchersBase = launchers.remote(ngpus=1),  # noqa B008
                 log_path: str = None, openai_api: Optional[bool] = None, **kw):
        self.launcher_list, launcher = reallocate_launcher(launcher)
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
            'pipeline-parallel-size': 1,
            'max-num-batched-tokens': 64000,
        })
        if openai_api is None: openai_api = lazyllm.config['openai_api']
        self._vllm_cmd = 'vllm.entrypoints.openai.api_server' if openai_api else 'vllm.entrypoints.api_server'
        self._openai_api = openai_api
        self.trust_remote_code = trust_remote_code
        self.kw.update(**{key: kw[key] for key in self.optional_keys if key in kw})
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'vllm') if log_path else None
        if self.launcher_list:
            ray_launcher = [Distributed(launcher=launcher) for launcher in self.launcher_list]
            parall_launcher = [lazyllm.pipeline(sleep_moment, launcher) for launcher in ray_launcher[1:]]
            self._prepare_deploy = lazyllm.pipeline(
                ray_launcher[0], post_action=(lazyllm.parallel(*parall_launcher) if len(parall_launcher) else None))

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        """Build the command to launch the vLLM inference service.

This method validates the model path and constructs an executable command string based on current configuration. In distributed mode, it will also prepend the ray cluster start command.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Fallback base model path if finetuned_model is invalid.
    master_ip (str): IP address of the master node in a distributed setup.

**Returns:**

- LazyLLMCMD: The command object with shell instruction, return value handler, and health checker.
"""
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                            f'base_model({base_model}) will be used')
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            cmd = ''
            if self.launcher_list:
                cmd += f'ray start --address="{master_ip}" && '
            cmd += f'{sys.executable} -m {self._vllm_cmd} --model {finetuned_model} '
            if self._openai_api: cmd += '--served-model-name lazyllm '
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust-remote-code '
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl,
                          checkf=(verify_vllm_openai_func if self._openai_api else verify_fastapi_func))

    def geturl(self, job=None):
        """Get the inference service URL for the vLLM deployment.

Depending on the execution mode (Display or actual deployment), this method returns the appropriate URL for accessing the model's generate endpoint.

Args:
    job (Job, optional): Deployment job object. Defaults to the module's associated job.

**Returns:**

- str: The HTTP URL for inference service.
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/generate'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}' + (
                '/v1/' if self._openai_api else '/generate')

    @staticmethod
    def extract_result(x, inputs):
        """Extracts the inference result from the JSON string returned by the VLLM service.

Args:
    x (str): Raw JSON string returned from the VLLM service.
    inputs (Any): Input arguments (not used here, kept for interface consistency).

**Returns:**

- str: The generated text result from the model.
"""
        return json.loads(x)['text'][0]
