from typing import Any
from ..core import LLMBase
from lazyllm import launchers, flows
import random


class LazyLLMDeployBase(LLMBase):

    def __init__(self, *, launcher=launchers.slurm()):
        super().__init__(launcher=launcher)


class DummyDeploy(LazyLLMDeployBase, flows.NamedPipeline):
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