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
    """The ``LMDeploy`` class, a subclass of ``LazyLLMDeployBase``,  
leverages [LMDeploy](https://github.com/InternLM/lmdeploy) to launch and manage large language model inference services.

Args:
    launcher (Optional[lazyllm.launcher]): The service launcher, defaults to ``launchers.remote(ngpus=1)``.  
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.  
    log_path (Optional[str]): Path to store logs, defaults to ``None``.  
    **kw: Keyword arguments used to update the default deployment configuration. No extra arguments beyond those listed below are allowed.  

Keyword Args:
    tp (int): Tensor parallelism factor, defaults to ``1``.  
    server-name (str): The IP address on which the service listens, defaults to ``0.0.0.0``.  
    server-port (Optional[int]): Port number for the service. Defaults to ``None``; in this case, a random port between 30000–40000 will be assigned.  
    max-batch-size (int): Maximum batch size, defaults to ``128``.  
    chat-template (Optional[str]): Path to the chat template file. If the model is not a vision-language model and no template is specified, a default template will be used.  
    eager-mode (bool): Whether to enable eager mode, controlled by the environment variable ``LMDEPLOY_EAGER_MODE``, defaults to ``False``.  


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
        'image_url': None,
        'session_id': -1,
        'interactive_mode': False,
        'stream': False,
        'stop': None,
        'request_output_len': None,
        'top_p': 0.8,
        'top_k': 40,
        'temperature': 0.8,
        'repetition_penalty': 1,
        'max_new_tokens': 4096,
        'ignore_eos': False,
        'skip_special_tokens': True,
        'cancel': False,
        'adapter_name': None
    }
    auto_map = {
        'port': 'server-port',
        'host': 'server-name',
        'max_batch_size': 'max-batch-size',
        'chat_template': 'chat-template',
    }
    stream_parse_parameters = {'delimiter': b'\n'}

    def __init__(self, launcher=launchers.remote(ngpus=1), trust_remote_code=True, log_path=None, **kw):  # noqa B008
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'server-name': '0.0.0.0',
            'server-port': None,
            'tp': 1,
            'max-batch-size': 128,
            'chat-template': None,
        })
        self.kw.check_and_update(kw)
        self._trust_remote_code = trust_remote_code
        self.random_port = False if 'server-port' in kw and kw['server-port'] else True
        self.temp_folder = make_log_dir(log_path, 'lmdeploy') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        """This method generates the command to start the LMDeploy service.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Path to the base model, used when finetuned_model is invalid.

**Returns:**

- LazyLLMCMD: A LazyLLMCMD object containing the startup command.
"""
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                            f'base_model({base_model}) will be used')
            finetuned_model = base_model

        model_type = ModelManager.get_model_type(base_model or finetuned_model)
        if model_type == 'vlm':
            self.kw.pop('chat-template')
        else:
            if not self.kw['chat-template'] and 'vl' not in finetuned_model and 'lava' not in finetuned_model:
                self.kw['chat-template'] = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'lmdeploy', 'chat_template.json')
            else:
                self.kw.pop('chat-template')

        def impl():
            if self.random_port:
                self.kw['server-port'] = random.randint(30000, 40000)
            cmd = f'lmdeploy serve api_server {finetuned_model} '

            if importlib.util.find_spec('torch_npu') is not None: cmd += '--device ascend '
            if config['lmdeploy_eager_mode']: cmd += '--eager-mode '
            cmd += self.kw.parse_kwargs()
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        """Get the URL address of the LMDeploy service.

Args:
    job (optional): Job object, defaults to None, in which case self.job is used.

**Returns:**

- str: The service URL address in the format "http://{ip}:{port}/v1/chat/interactive".
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/v1/chat/interactive'
        else:
            return f'http://{job.get_jobip()}:{self.kw["server-port"]}/v1/chat/interactive'

    @staticmethod
    def extract_result(x, inputs):
        """Parses the model inference result and extracts the text output from a JSON response string.

Args:
    x (str): JSON-formatted string returned by the model.  
    inputs (dict): The original input data (not directly used, reserved for interface compatibility).  

**Returns:**

- str: The text result extracted from the response.  
"""
        return json.loads(x)['text']
