import time
from dataclasses import dataclass
from typing import Callable, Any, Dict
from ..core import ComponentBase
import lazyllm
from lazyllm import launchers, flows, LOG
import random


@dataclass
class KwMapItem:
    new_key: str
    type_func: Callable[[Any], Any]
    
class LazyLLMDeployBase(ComponentBase):
    keys_name_handle = None
    message_format = None
    default_headers = {'Content-Type': 'application/json'}
    stream_url_suffix = ''
    stream_parse_parameters = {}

    @staticmethod
    def extract_result(output, inputs):
        return output

    def __init__(self, *, launcher=launchers.remote()):
        super().__init__(launcher=launcher)

    def kw_map_for_framework(self, kw: Dict[str, Any], kw_map: Dict[str, KwMapItem]) -> Dict[str, Any]:
        result = {}
        for k, v in kw.items():
            kw_item = kw_map.get(k)
            if kw_item:
                try:
                    result[kw_item.new_key] = kw_item.type_func(v)
                except (TypeError, ValueError) as e:
                    LOG.warning(f"Type conversion error for key '{k}': {e}, using original value")
                    result[kw_item.new_key] = v
        return result


class DummyDeploy(LazyLLMDeployBase, flows.Pipeline):
    keys_name_handle = {'inputs': 'inputs'}
    message_format = {
        'inputs': '',
        'parameters': {
            'do_sample': False,
            'temperature': 0.1,
        }
    }

    def __init__(self, launcher=launchers.remote(sync=False), *, stream=False, **kw):
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
