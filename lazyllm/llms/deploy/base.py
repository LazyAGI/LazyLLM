import json
import time
import requests
from ..core import LLMBase
import lazyllm
from lazyllm import launchers, flows, LazyLLMCMD, timeout
import random


class LazyLLMDeployBase(LLMBase):

    def __init__(self, *, launcher=launchers.slurm()):
        super().__init__(launcher=launcher)


class DummyDeploy(LazyLLMDeployBase, flows.NamedPipeline):
    input_key_name = 'inputs'
    default_headers = {'Content-Type': 'application/json'}
    message_formate = None
    
    def __init__(self, launcher=launchers.slurm(sync=False), **kw):
        super().__init__(launcher=launcher)
        def func():
            def impl(x):
                print(f'input is {x}')
                return f'reply for {x}'
            return impl
        flows.Pipeline.__init__(self, func,
            deploy.RelayServer(port=random.randint(30000, 40000), launcher=launcher))

    def __call__(self, *args):
        url = flows.NamedPipeline.__call__(self)
        print(f'dummy deploy url is : {url}')
        return url

    def __repr__(self):
        return flows.NamedPipeline.__repr__(self)


def verify_fastapi_func(job):
    while True:
        line = job.queue.get()
        if line.startswith('ERROR:'):
            print("Capture error message: ", line, "\n\n")
            return False, line
        elif 'Uvicorn running on' in line:
            print("Capture startup message:   ",line)
            break
        if job.status == lazyllm.launchers.status.Failed:
            return False, 'Service Startup Failed.'
    return True, line


def restart_service(job, port, headers, data, build_cmd=None, verify_func=verify_fastapi_func, count=0):
    assert count < 5, 'The service failed to restart 5 times.'
    
    if lazyllm.mode == lazyllm.Mode.Display:
        return f'http://{job.name}:{port}/generate'
    status = launchers.status
    with timeout(3600, msg='Launch failed: No computing resources are available.'):
        while job.status in (status.TBSubmitted, status.InQueue, status.Pending):
            time.sleep(2)
    
    assert job.status == lazyllm.launchers.status.Running, 'Job failed'
    url = f'http://{job.get_jobip()}:{port}/generate'

    with timeout(240, msg='Service encountered an unknown exception.'):
        verify_res, verify_str = verify_func(job)
        data = 'Hello world~' if not data else data
        if verify_res:
            _ = requests.post(url, headers=headers, data=json.dumps(data))
            job.return_value = url
            return url
        elif build_cmd:
            cmd, port = build_cmd()
            job.cmd = LazyLLMCMD(cmd=cmd)
            job.restart()
            return restart_service(job, port, headers, data, build_cmd, verify_func, count+1)
        else:
            raise verify_str
