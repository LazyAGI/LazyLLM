from typing import Any
from ..core import LLMBase
from lazyllm import launchers


class LazyLLMDeployBase(LLMBase):

    def __init__(self, *, launcher=launchers.slurm()):
        super().__init__(launcher=launcher)