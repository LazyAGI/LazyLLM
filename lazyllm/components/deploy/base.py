import time
from ..core import ComponentBase
import lazyllm
from lazyllm import launchers, flows, LOG
import random


class LazyLLMDeployBase(ComponentBase):

    def __init__(self, *, launcher=launchers.remote()):
        super().__init__(launcher=launcher)


class DummyDeploy(LazyLLMDeployBase, flows.Pipeline):
    keys_name_handle = {'inputs': 'inputs'}
    default_headers = {'Content-Type': 'application/json'}
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
