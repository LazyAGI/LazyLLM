import time
import random
from queue import Empty
from typing import Callable

from ..core import ComponentBase
import lazyllm
from lazyllm import launchers, flows, LOG
from ...components.utils.file_operate import _image_to_base64, _audio_to_base64, ocr_to_base64

lazyllm.config.add('openai_api', bool, False, 'OPENAI_API', description='Whether to use OpenAI API for vllm deployer.')


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

def verify_func_factory(error_message: str, running_message: str,  # noqa: C901
                        err_judge: Callable = lambda syb, msg: msg.lstrip().startswith(syb),
                        run_judge: Callable = lambda syb, msg: syb in msg):
    def _hit(symbols, msg, judge):
        return judge(symbols, msg) if isinstance(symbols, str) else any([judge(s, msg) for s in symbols])

    def verify_func(job):
        begin_time = time.time()
        while True:
            try:
                line = job.queue.get(timeout=3)
            except Empty:
                line = ''
                status = job.status
                if status == lazyllm.launchers.status.Failed:
                    LOG.error('[Verify] Service Startup Failed, '
                              'use `export LAZYLLM_EXPECTED_LOG_MODULES=all` for more logs')
                    return False
                LOG.debug(f'[Verify] Timeout when getting log line and current service status: {status}.')
            if _hit(error_message, line, err_judge):
                LOG.error(f'[Verify] Capture error message: {line} \n\n '
                          ', use `export LAZYLLM_EXPECTED_LOG_MODULES=all` for more logs')
                return False
            elif _hit(running_message, line, run_judge):
                LOG.info(f'[Verify] Capture startup message: {line}', name='launcher')
                LOG.success(f'job `{str(job._fixed_cmd).strip()}` executed successfully!')
                break
            if time.time() - begin_time > 600:
                LOG.error('[Verify] Service Startup Timeout, '
                          'use `export LAZYLLM_EXPECTED_LOG_MODULES=all` for more logs')
                return False
        return True
    return verify_func

verify_fastapi_func = verify_func_factory('ERROR:', 'Uvicorn running on')
verify_ray_func = verify_func_factory(['ray.exceptions.RayTaskError', 'Traceback (most recent call last)'],
                                      'Deployed app \'default\' successfully', err_judge=lambda syb, msg: syb in msg)
