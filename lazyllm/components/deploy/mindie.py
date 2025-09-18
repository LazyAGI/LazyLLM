import os
import json
import random
import shutil

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_func_factory
from .utils import get_log_path, make_log_dir

lazyllm.config.add('mindie_home', str, '', 'MINDIE_HOME')

verify_fastapi_func = verify_func_factory(error_message='Service Startup Failed',
                                          running_message='Daemon start success')
class Mindie(LazyLLMDeployBase):
    """This class is a subclass of ``LazyLLMDeployBase``, designed for deploying and managing the MindIE large language model inference service. It encapsulates the full workflow including configuration generation, process launching, and API interaction for the MindIE service.

Args:
    trust_remote_code (bool): Whether to trust remote code (e.g., from HuggingFace models). Default is ``True``.
    launcher: Instance of the task launcher. Default is ``launchers.remote()``.
    log_path (str): Path to save logs. If ``None``, logs will not be saved.
    **kw: Other configuration parameters.

Keyword Args: 
            npuDeviceIds: List of NPU device IDs (e.g., ``[[0,1]]`` indicates using 2 devices)
            worldSize: Model parallelism size
            port: Service port (set to ``'auto'`` for auto-assignment between 30000–40000)
            maxSeqLen: Maximum sequence length
            maxInputTokenLen: Maximum number of tokens per input
            maxPrefillTokens: Maximum number of prefill tokens
            config: Custom configuration file

Notes:
    You must set the environment variable ``LAZYLLM_MINDIE_HOME`` to point to the MindIE installation directory. 
    If ``finetuned_model`` is not specified or the path is invalid, it will automatically fall back to ``base_model``.


Examples:
    >>> import lazyllm
    >>> from lazyllm.components.deploy import Mindie            
    >>> deployer = Mindie(
    ...     port=30000,
    ...     launcher=lazyllm.launchers.remote(),
    ...     max_seq_len=32000,
    ...     log_path="/path/to/logs"
    ... )
    >>> cmd = deployer.cmd(
    ...     finetuned_model="/path/to/finetuned_model",
    ...     base_model="/path/to/base_model")
    >>> print("Service URL:", cmd.geturl())
    
    """
    keys_name_handle = {
        'inputs': 'prompt',
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'max_tokens': 4096,
        'presence_penalty': 1.03,
        'frequency_penalty': 1.0,
        'temperature': 0.5,
        'top_p': 0.95
    }
    auto_map = {
        'port': int,
        'tp': ('world_size', int),
        'max_input_token_len': ('maxInputTokenLen', int),
        'max_prefill_tokens': ('maxPrefillTokens', int),
        'max_seq_len': ('maxSeqLen', int)
    }

    def __init__(self, trust_remote_code=True, launcher=launchers.remote(), log_path=None, **kw):  # noqa B008
        super().__init__(launcher=launcher)
        assert lazyllm.config['mindie_home'], 'Ensure you have installed MindIE and \
                                  "export LAZYLLM_MINDIE_HOME=/path/to/mindie/latest"'
        self.mindie_home = lazyllm.config['mindie_home']
        self.mindie_config_path = os.path.join(self.mindie_home, 'mindie-service/conf/config.json')
        self.backup_path = self.mindie_config_path + '.backup'
        self.custom_config = kw.pop('config', None)
        self.kw = ArgsDict({
            'npuDeviceIds': [[0]],
            'worldSize': 1,
            'port': 'auto',
            'host': '0.0.0.0',
            'maxSeqLen': 64000,
            'maxInputTokenLen': 4096,
            'maxPrefillTokens': 8192,
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)
        self.kw['npuDeviceIds'] = [[i for i in range(self.kw.get('worldSize', 1))]]
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'mindie') if log_path else None

        if self.custom_config:
            self.config_dict = (ArgsDict(self.load_config(self.custom_config))
                                if isinstance(self.custom_config, str) else ArgsDict(self.custom_config))
            self.kw['host'] = self.config_dict['ServerConfig']['ipAddress']
            self.kw['port'] = self.config_dict['ServerConfig']['port']
        else:
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindie', 'config.json')
            self.config_dict = ArgsDict(self.load_config(default_config_path))

    def __del__(self):
        if hasattr(self, 'backup_path') and os.path.isfile(self.backup_path):
            shutil.copy2(self.backup_path, self.mindie_config_path)

    def load_config(self, config_path):
        """Loads and parses the MindIE configuration file.

Args:
    config_path (str): Path to the JSON configuration file

**Returns:**

- dict: Parsed configuration dictionary

Notes:
    - Handles both default and custom configuration files
    - Uses JSON format for configuration
    - Creates backup of original config before modification
"""
        with open(config_path, 'r') as file:
            config_dict = json.load(file)
        return config_dict

    def save_config(self):
        """Saves the current configuration to file.

Notes:
    - Automatically creates backup of existing config
    - Writes to the standard MindIE config location
    - Uses JSON format with proper indentation
    - Called automatically during deployment
"""
        if os.path.isfile(self.mindie_config_path):
            shutil.copy2(self.mindie_config_path, self.backup_path)

        with open(self.mindie_config_path, 'w') as file:
            json.dump(self.config_dict, file)

    def update_config(self):
        """Updates the configuration dictionary with current settings.

Notes:
    - Handles multiple configuration sections:
        - Model deployment parameters
        - Server settings
        - Scheduling parameters
"""
        backend_config = self.config_dict['BackendConfig']
        backend_config['npuDeviceIds'] = self.kw['npuDeviceIds']
        model_config = {
            'modelName': self.finetuned_model.split('/')[-1],
            'modelWeightPath': self.finetuned_model,
            'worldSize': self.kw['worldSize'],
            'trust_remote_code': self.trust_remote_code
        }
        backend_config['ModelDeployConfig']['ModelConfig'][0].update(model_config)
        backend_config['ModelDeployConfig']['maxSeqLen'] = self.kw['maxSeqLen']
        backend_config['ModelDeployConfig']['maxInputTokenLen'] = self.kw['maxInputTokenLen']
        backend_config['ScheduleConfig']['maxPrefillTokens'] = self.kw['maxPrefillTokens']
        self.config_dict['BackendConfig'] = backend_config
        if self.kw['host'] != '0.0.0.0':
            self.config_dict['ServerConfig']['ipAddress'] = self.kw['host']
        self.config_dict['ServerConfig']['port'] = self.kw['port']

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        """Generates the command to start the MindIE service.

Args:
    finetuned_model (str): Path to the fine-tuned model
    base_model (str): Path to the base model (fallback if finetuned_model is invalid)
    master_ip (str): Master node IP address (currently unused)

**Returns:**

- LazyLLMCMD: Command object for starting the service

Notes:
    - Automatically handles model path validation
    - Updates configuration before service start
    - Supports random port allocation when configured
"""
        if self.custom_config is None:
            self.finetuned_model = finetuned_model
            if finetuned_model or base_model:
                if not os.path.exists(finetuned_model) or \
                    not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                            for filename in os.listdir(finetuned_model)):
                    if not finetuned_model:
                        LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                                    f'base_model({base_model}) will be used')
                    self.finetuned_model = base_model

            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            self.update_config()

        self.save_config()

        def impl():
            cmd = f'{os.path.join(self.mindie_home, "mindie-service/bin/mindieservice_daemon")}'
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        """Gets the service URL after deployment.

Args:
    job: Job object (optional, defaults to self.job)

**Returns:**

- str: The generate endpoint URL

Notes:
    - Returns different formats based on display mode
    - Includes port number from configuration
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'
        else:
            LOG.info(f'MindIE Server running on http://{job.get_jobip()}:{self.kw["port"]}')
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'

    @staticmethod
    def extract_result(x, inputs):
        """Extracts the generated text from the API response.

Args:
    x: Raw API response
    inputs: Original inputs (unused)

**Returns:**

- str: The generated text

Notes:
    - Parses JSON response
    - Returns first text entry from response
"""
        return json.loads(x)['text'][0]
