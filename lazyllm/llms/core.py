from lazyllm import LazyLLMRegisterMetaClass
from lazyllm import launchers
import re
import time

class LLMBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *, launcher=launchers.empty):
        self.launcher = launcher()

    def apply():
        raise NotImplementedError('please implement function \'apply\'')

    def cmd() -> str:
        raise NotImplementedError('please implement function \'cmd\'')

    def __call__(self, *args, **kw):
        if getattr(self.__class__, 'apply') is not getattr(__class__, 'apply'):
            assert getattr(self.__class__, 'cmd') is getattr(__class__, 'cmd'), (
                'Cannot override \'cmd\' and \'apply\' in the same class')
            return self.launcher.launch(self.apply, *args, **kw)
        else:
            cmd = self.cmd(*args, **kw)
            return self.launcher.launch(cmd)


reg_template = '''\
class {name}{base}(LazyLLMRegisterMetaClass.all_groups[\'{base}\'.lower()]):
    pass
'''


def register(cls, templates=reg_template, cmd=False):
    cls = cls.__name__ if isinstance(cls, type) else cls
    cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2] \
        if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls
    def impl(func):
        func_name = func.__name__
        exec(templates.format(name=func_name, base=cls.capitalize()))
        # 'func' cannot be recognized by exec, so we use 'setattr' instead 
        f = LazyLLMRegisterMetaClass.all_clses[cls.lower()].__getattr__(func_name)
        if cmd:
            f.cmd = lambda _, cmd : func(cmd) 
        else:
            f.apply = lambda _, *args, **kw : func(*args, **kw) 
    return impl


