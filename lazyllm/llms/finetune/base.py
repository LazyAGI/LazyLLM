from lazyllm import launchers
from ..core import LLMBase


class LazyLLMFinetuneBase(LLMBase):
    def __init__(self, base_model, target_path, *, launcher=launchers.empty):
        super().__init__(launcher=launcher)
        self.base_model = base_model
        self.target_path = target_path

    def cmd() -> str:
        raise NotImplementedError('please implement function \'cmd\'')

    def __call__(self, trainset, evalset):
        super().__call__(trainset, evalset)
        return self.base_model, self.target_path