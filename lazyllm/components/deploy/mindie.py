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

    def __init__(self, trust_remote_code=True, launcher=launchers.remote(), stream=False, log_path=None, **kw):
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
            'maxInputTokenLen': 8192
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'mindie') if log_path else None

        if self.custom_config:
            self.config_dict = (ArgsDict(self.load_config(self.custom_config))
                                if isinstance(self.custom_config, str) else ArgsDict(self.custom_config))
            self.kw['host'] = self.config_dict["ServerConfig"]["ipAddress"]
            self.kw['port'] = self.config_dict["ServerConfig"]["port"]
        else:
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindie', 'config.json')
            self.config_dict = ArgsDict(self.load_config(default_config_path))

    def __del__(self):
        if hasattr(self, 'backup_path') and os.path.isfile(self.backup_path):
            shutil.copy2(self.backup_path, self.mindie_config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config_dict = json.load(file)
        return config_dict

    def save_config(self):
        if os.path.isfile(self.mindie_config_path):
            shutil.copy2(self.mindie_config_path, self.backup_path)

        with open(self.mindie_config_path, 'w') as file:
            json.dump(self.config_dict, file)

    def update_config(self):
        backend_config = self.config_dict["BackendConfig"]
        backend_config["npuDeviceIds"] = self.kw["npuDeviceIds"]
        model_config = {
            "modelName": self.finetuned_model.split('/')[-1],
            "modelWeightPath": self.finetuned_model,
            "worldSize": self.kw["worldSize"],
            "trust_remote_code": self.trust_remote_code
        }
        backend_config["ModelDeployConfig"]["ModelConfig"][0].update(model_config)
        backend_config["ModelDeployConfig"]["maxSeqLen"] = self.kw["maxSeqLen"]
        backend_config["ModelDeployConfig"]["maxInputTokenLen"] = self.kw["maxInputTokenLen"]
        self.config_dict["BackendConfig"] = backend_config
        if self.kw["host"] != '0.0.0.0':
            self.config_dict["ServerConfig"]["ipAddress"] = self.kw["host"]
        self.config_dict["ServerConfig"]["port"] = self.kw["port"]

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        if self.custom_config is None:
            self.finetuned_model = finetuned_model
            if finetuned_model or base_model:
                if not os.path.exists(finetuned_model) or \
                    not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                            for filename in os.listdir(finetuned_model)):
                    if not finetuned_model:
                        LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                                    f"base_model({base_model}) will be used")
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
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'
        else:
            LOG.info(f"MindIE Server running on http://{job.get_jobip()}:{self.kw['port']}")
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'

    @staticmethod
    def extract_result(x, inputs):
        return json.loads(x)['text'][0]

    @staticmethod
    def stream_url_suffix():
        return ''
