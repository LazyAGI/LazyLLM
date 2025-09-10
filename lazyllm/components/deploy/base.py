import time
import os
import yaml
import requests

from ..core import ComponentBase
import lazyllm
from lazyllm import launchers, flows, LOG
from ...components.utils.file_operate import _image_to_base64, _audio_to_base64, ocr_to_base64
import random


trainable_module_config_map = {}
lazyllm.config.add('trainable_module_config_map_path', str, '', 'TRAINABLE_MODULE_CONFIG_MAP_PATH')
if os.path.exists(lazyllm.config['trainable_module_config_map_path']):
    trainable_module_config_map = yaml.safe_load(open(lazyllm.config['trainable_module_config_map_path'], 'r'))
lazyllm.config.add('openai_api', bool, False, 'OPENAI_API')

class LazyLLMDeployBase(ComponentBase):
    keys_name_handle = None
    message_format = None
    default_headers = {'Content-Type': 'application/json'}
    stream_url_suffix = ''
    stream_parse_parameters = {}

    encoder_map = dict(image=_image_to_base64, audio=_audio_to_base64, ocr_files=ocr_to_base64)

    @staticmethod
    def extract_result(output, inputs):
        return output

    def __init__(self, *, launcher=launchers.remote()):  # noqa B008
        super().__init__(launcher=launcher)

    def _get_available_url(self, base_model: str):
        if base_model not in trainable_module_config_map:
            return False
        model_configs = trainable_module_config_map[base_model]
        for model_config in model_configs:
            url = model_config.get('url')
            deploy_config = model_config.get('deploy_config')
            framework = deploy_config.get('framework')
            check_status = True
            check_status &= framework and url and framework.lower() == self.__class__.__name__.lower()
            for key, value in self.kw.items():
                check_status &= deploy_config.get(key) == value
            check_status &= set(self.store_true_keys).issubset(set(deploy_config.get('lazyllm-store-true-keys', [])))

            try:
                requests.get(url, timeout=3)
                check_status &= True
            except Exception:
                LOG.info(f"url not available: {url}")
                check_status &= False

            if check_status:
                return url
        return None

    def _add_config(self, base_model: str, url: str):
        framework = self.__class__.__name__.lower()
        model_config = {
            'url': url,
            'deploy_config': {
                'framework': framework,
                'lazyllm-store-true-keys': self.store_true_keys,
                **self.kw
            }
        }
        trainable_module_config_map.setdefault(base_model, [])
        trainable_module_config_map[base_model].append(model_config)

    def __call__(self, *args, **kw):
        assert len(args) >= 2 and os.path.exists(args[1]), (
            'Deploy component requires at least two arguments: base_model and target_path'
        )
        base_model = os.path.basename(args[1])
        if not (url := self._get_available_url(base_model)):
            url = super().__call__(*args, **kw)
            self._add_config(base_model, url)
        return url


class DummyDeploy(LazyLLMDeployBase, flows.Pipeline):
    keys_name_handle = {'inputs': 'inputs'}
    message_format = {
        'inputs': '',
        'parameters': {
            'do_sample': False,
            'temperature': 0.1,
        }
    }

    def __init__(self, launcher=launchers.remote(sync=False), *, stream=False, **kw):  # noqa B008
        super().__init__(launcher=launcher)

        def func():

            def impl(x):
                LOG.info(f'input is {x["inputs"]}, parameters is {x["parameters"]}')
                return f'reply for {x["inputs"]}, and parameters is {x["parameters"]}'

            def impl_stream(x):
                for s in ['reply', ' for', f' {x["inputs"]}', ', and',
                          ' parameters', ' is', f' {x["parameters"]}']:
                    yield s
                    time.sleep(0.2)
            return impl_stream if stream else impl
        flows.Pipeline.__init__(self, func,
                                lazyllm.deploy.RelayServer(port=random.randint(30000, 40000), launcher=launcher))

    def __call__(self, *args):
        url = flows.Pipeline.__call__(self)
        LOG.info(f'dummy deploy url is : {url}')
        return url

    def __repr__(self):
        return flows.Pipeline.__repr__(self)

def verify_func_factory(error_message='ERROR:',
                        running_message='Uvicorn running on'):
    def verify_func(job):
        while True:
            line = job.queue.get()
            if line.startswith(error_message):
                LOG.error(f"Capture error message: {line} \n\n")
                return False
            elif running_message in line:
                LOG.info(f"Capture startup message: {line}")
                break
            if job.status == lazyllm.launchers.status.Failed:
                LOG.error("Service Startup Failed.")
                return False
        return True
    return verify_func

verify_fastapi_func = verify_func_factory()
