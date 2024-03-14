from lazyllm import launchers, package
from ..core import LLMBase


class LazyLLMFinetuneBase(LLMBase):
    def __init__(self, base_model, target_path, *, launcher=launchers.slurm()):
        super().__init__(launcher=launcher)
        self.base_model = base_model
        self.target_path = target_path
        self.merge_path = None

    def cmd(*args, **kw) -> str:
        raise NotImplementedError('please implement function \'cmd\'')

    def __call__(self, *args, **kw):
        super().__call__(*args, **kw)
        if self.merge_path:
            return self.merge_path
        else:
            return self.target_path

class ArgsDict(dict):
    def __init__(self, *args, **kwargs):
        super(ArgsDict, self).__init__(*args, **kwargs)

    def check(self, kw):
        assert set(kw.keys()).issubset(set(self))
        self.update(kw)

    def parse_kwargs(self):
        string = ' '.join(f'--{k}={v}' if type(v) is not str else f'--{k}=\"{v}\"' for k, v in self.items())
        return string


class DummyFinetune(LazyLLMFinetuneBase):
    def __init__(self, base_model='base', target_path='target', *, launcher=launchers.slurm()):
        super().__init__(base_model, target_path, launcher=launchers.empty)

    def cmd(*args, **kw) -> str:
        return 'echo dummy finetune!'
