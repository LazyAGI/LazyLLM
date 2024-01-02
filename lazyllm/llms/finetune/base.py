from lazyllm import launchers, package
from ..core import LLMBase


class LazyLLMFinetuneBase(LLMBase):
    def __init__(self, base_model, target_path, *, launcher=launchers.slurm()):
        super().__init__(launcher=launcher)
        self.base_model = base_model
        self.target_path = target_path

    def cmd(*args, **kw) -> str:
        raise NotImplementedError('please implement function \'cmd\'')

    def __call__(self, *args, **kw):
        super().__call__(*args, **kw)
        return package([self.base_model, self.target_path])