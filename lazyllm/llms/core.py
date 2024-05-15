import lazyllm
from lazyllm import LazyLLMRegisterMetaClass
from lazyllm import LazyLLMCMD, ReadOnlyWrapper
from lazyllm import launchers, LazyLLMLaunchersBase
from typing import Union

class ComponentBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *, launcher=launchers.empty()):
        self._llm_name = None
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

    @property
    def name(self): return self._llm_name
    @name.setter
    def name(self, name): self._llm_name = name

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
        return getattr(self.__class__, f) is not getattr(__class__, f) or \
            getattr(self.__class__, '__reg_overwrite__', None) == f

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
        return lazyllm.make_repr('lazyllm.llm.' + self.__class__._lazy_llm_group,
                                 self.__class__.__name__, name=self.name)


register = lazyllm.Register(ComponentBase, ['apply', 'cmd'])
