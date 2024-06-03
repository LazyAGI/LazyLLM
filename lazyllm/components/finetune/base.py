from lazyllm import launchers
from ..core import ComponentBase


class LazyLLMFinetuneBase(ComponentBase):
    __reg_overwrite__ = 'cmd'

    def __init__(self, base_model, target_path, *, launcher=launchers.remote()):
        super().__init__(launcher=launcher)
        self.base_model = base_model
        self.target_path = target_path
        self.merge_path = None

    def __call__(self, *args, **kw):
        super().__call__(*args, **kw)
        if self.merge_path:
            return self.merge_path
        else:
            return self.target_path


class DummyFinetune(LazyLLMFinetuneBase):
    def __init__(self, base_model='base', target_path='target', *, launcher=launchers.remote(), **kw):
        super().__init__(base_model, target_path, launcher=launchers.empty)
        self.kw = kw

    def cmd(self, *args, **kw) -> str:
        return f'echo \'dummy finetune!, and init-args is {self.kw}\''
