import lazyllm
from lazyllm import LazyLLMRegisterMetaClass
from lazyllm import LazyLLMCMD, ReadOnlyWrapper
from lazyllm import launchers, LazyLLMLaunchersBase
from typing import Union

class ComponentBase(object, metaclass=LazyLLMRegisterMetaClass):
    """Base class for components, providing a unified interface and basic implementation to facilitate creation of various components.  
Components execute tasks via a specified launcher and support custom task execution logic.

Args:
    launcher (LazyLLMLaunchersBase or type, optional): Launcher instance or launcher class used by the component, defaults to empty launcher.


Examples:
    >>> from lazyllm.components.core import ComponentBase
    >>> class MyComponent(ComponentBase):
    ...     def apply(self, x):
    ...         return x * 2
    >>> comp = MyComponent()
    >>> comp.name = "ExampleComponent"
    >>> print(comp.name)
    ExampleComponent
    >>> result = comp(10)
    >>> print(result)
    20
    >>> print(comp.apply(5))
    10
    """
    def __init__(self, *, launcher=launchers.empty()):  # noqa B008
        self._llm_name = None
        self.job = ReadOnlyWrapper()
        if isinstance(launcher, LazyLLMLaunchersBase):
            self._launcher = launcher
        elif isinstance(launcher, type) and issubclass(launcher, LazyLLMLaunchersBase):
            self._launcher = launcher()
        else:
            raise RuntimeError('Invalid launcher given:', launcher)

    def apply():
        """Core execution method of the component, to be implemented by subclasses.  
Defines the specific business logic or task execution steps of the component.

**Note:**  
If this method is overridden by the subclass, it will be called when the component is invoked.
"""
        raise NotImplementedError('please implement function \'apply\'')

    def cmd(self, *args, **kw) -> Union[str, tuple, list]:
        """Generates the execution command of the component, to be implemented by subclasses.  
The returned command can be a string, tuple, or list, representing the instruction to execute the task.

**Note:**  
If the `apply` method is not overridden, this command will be used to create a job for the launcher to run.
"""
        raise NotImplementedError('please implement function \'cmd\'')

    @property
    def name(self): return self._llm_name
    @name.setter
    def name(self, name): self._llm_name = name

    @property
    def launcher(self): return self._launcher

    def _get_job_with_cmd(self, *args, **kw):
        cmd = self.cmd(*args, **kw)
        cmd = cmd if isinstance(cmd, LazyLLMCMD) else LazyLLMCMD(cmd)
        return self._launcher.makejob(cmd=cmd)

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f) or \
            getattr(self.__class__, '__reg_overwrite__', None) == f

    def __call__(self, *args, **kw):
        if self._overwrote('apply'):
            assert not self._overwrote('cmd'), (
                'Cannot overwrite \'cmd\' and \'apply\' in the same class')
            assert isinstance(self._launcher, launchers.Empty), 'Please use EmptyLauncher instead.'
            return self._launcher.launch(self.apply, *args, **kw)
        else:
            job = self._get_job_with_cmd(*args, **kw)
            self.job.set(job)
            return self._launcher.launch(job)

    def __repr__(self):
        return lazyllm.make_repr('lazyllm.llm.' + self.__class__._lazy_llm_group,
                                 self.__class__.__name__, name=self.name)


register = lazyllm.Register(ComponentBase, ['apply', 'cmd'])
