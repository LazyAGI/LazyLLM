from lazyllm import LazyLLMRegisterMetaClass
from lazyllm import launchers, LazyLLMLaunchersBase
import re
from typing import Union, Optional

class LLMBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *, launcher=launchers.empty()):
        if isinstance(launcher, LazyLLMLaunchersBase):
            self.launcher = launcher
        elif isinstance(launcher, type) and issubclass(launcher, LazyLLMLaunchersBase):
            self.launcher = launcher()
        else:
            raise RuntimeError('Invalid launcher given:', launcher)

    def apply():
        raise NotImplementedError('please implement function \'apply\'')

    def cmd(self, *args, **kw) -> Union[str, tuple, list]:
        raise NotImplementedError('please implement function \'cmd\'')

    def _get_slurm_job(self, *args, **kw):
        raise NotImplementedError('please implement function \'_get_slurm_job\'')

    def _get_sco_job(self, *args, **kw):
        raise NotImplementedError('please implement function \'_get_sco_job\'')

    def _get_job_with_cmd(self, *args, **kw):
        if isinstance(self.launcher, launchers.slurm) and self._overwrote('_get_slurm_job'):
            self._get_slurm_job(*args, **kw)
        elif isinstance(self.launcher, launchers.sco) and self._overwrote('_get_sco_job'):
            self._get_sco_job(*args, **kw)
        else:
            return self.launcher.makejob(cmd=self.cmd(*args, **kw))

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f)

    def __call__(self, *args, **kw):
        if self._overwrote('apply'):
            assert not self._overwrote('cmd'), (
                'Cannot overwrite \'cmd\' and \'apply\' in the same class')
            return self.launcher.launch(self.apply, *args, **kw)
        else:
            job = self._get_job_with_cmd(*args, **kw)
            return self.launcher.launch(job)


reg_template = '''\
class {name}{base}(LazyLLMRegisterMetaClass.all_groups[\'{base}\'.lower()]):
    pass
'''


def register(cls, templates=reg_template, cmd=None):
    cls = cls.__name__ if isinstance(cls, type) else cls
    cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2] \
        if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls
    base = LazyLLMRegisterMetaClass.all_groups[cls.lower()]
    assert issubclass(base, LLMBase)
    cmd = (base.cmd != LLMBase.cmd) if cmd is None else cmd

    def impl(func):
        func_name = func.__name__
        exec(templates.format(name=func_name, base=cls.capitalize()))
        # 'func' cannot be recognized by exec, so we use 'setattr' instead 
        f = LazyLLMRegisterMetaClass.all_clses[cls.lower()].__getattr__(func_name)
        setattr(f, 'cmd' if cmd else 'apply', lambda _, *args, **kw : func(*args, **kw))
        return func
    return impl


