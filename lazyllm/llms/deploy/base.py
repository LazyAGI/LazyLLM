from typing import Any
from ..core import LLMBase
from lazyllm import FlowBase, launchers


class LazyLLMDeployBase(LLMBase, FlowBase):

    def __init__(self, *, launcher=launchers.slurm()):
        super().__init__(launcher=launcher)

    def __setattr__(self, name: str, value):
        if isinstance(value, LazyLLMDeployBase):
            self.items.append(value)
            value._flow_name = name
        return super().__setattr__(name, value)
