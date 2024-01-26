from lazyllm import LazyLLMRegisterMetaClass, _get_base_cls_from_registry
from lazyllm import LazyLLMCMD, ReadOnlyWrapper
from lazyllm import launchers, LazyLLMLaunchersBase
import re
from typing import Union, Optional

class LLMBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *, launcher=launchers.empty()):
        self._flow_name = None
        self.job = ReadOnlyWrapper()
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
            return self._get_slurm_job(*args, **kw)
        elif isinstance(self.launcher, launchers.sco) and self._overwrote('_get_sco_job'):
            return self._get_sco_job(*args, **kw)
        else:
            cmd = self.cmd(*args, **kw)
            cmd = cmd if isinstance(cmd, LazyLLMCMD) else LazyLLMCMD(cmd)
            return self.launcher.makejob(cmd=cmd)

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f)

    def __call__(self, *args, **kw):
        if self._overwrote('apply'):
            assert not self._overwrote('cmd'), (
                'Cannot overwrite \'cmd\' and \'apply\' in the same class')
            return self.launcher.launch(self.apply, *args, **kw)
        else:
            job = self._get_job_with_cmd(*args, **kw)
            self.job.set(job)
            return self.launcher.launch(job)

    def __repr__(self):
        represention = 'lazyllm.llm.core.' + self.__class__._lazy_llm_group
        represention += '.' + self.__class__.__name__
        return f'<{represention}>'


reg_template = '''\
class {name}(LazyLLMRegisterMetaClass.all_clses[\'{base}\'.lower()].base):
    pass
'''

class Register(object):
    def __init__(self, template=reg_template):
        self.template = template

    def __call__(self, cls, *, cmd=None):
        cls = cls.__name__ if isinstance(cls, type) else cls
        cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2] \
            if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls
        base = _get_base_cls_from_registry(cls.lower())
        assert issubclass(base, LLMBase)
        cmd = (base.cmd != LLMBase.cmd) if cmd is None else cmd

        def impl(func):
            func_name = func.__name__
            exec(self.template.format(
                name=func_name+cls.split('.')[-1].capitalize(), base=cls))
            # 'func' cannot be recognized by exec, so we use 'setattr' instead 
            f = LazyLLMRegisterMetaClass.all_clses[cls.lower()].__getattr__(func_name)
            f.__name__ = func_name
            setattr(f, 'cmd' if cmd else 'apply', lambda _, *args, **kw : func(*args, **kw))
            return func
        return impl

    def cmd(self, cls):
        return self(cls, cmd=True)

    def exe(self, cls):
        return self(cls, cmd=False)

register = Register()
