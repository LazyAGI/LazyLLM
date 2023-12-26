from lazyllm import LazyLLMRegisterMetaClass
import re

class LLMBase(object, metaclass=LazyLLMRegisterMetaClass):
    def apply():
        raise NotImplementedError('please implement function \'apply\'')

    def __call__(self, *args, **kw):
        return self.apply()

reg_template = '''\
class {name}{base}(LazyLLMRegisterMetaClass.all_groups[\'{base}\'.lower()]):
    pass
'''


def register(cls, templates=reg_template):
    cls = cls.__name__ if isinstance(cls, type) else cls
    cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2] \
        if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls
    def impl(func):
        func_name = func.__name__
        exec(templates.format(name=func_name, base=cls.capitalize()))
        # 'func' cannot be recognized by exec, so we use 'setattr' instead 
        LazyLLMRegisterMetaClass.all_clses[cls.lower()].__getattr__(func_name).apply = (
            lambda _, *args, **kw : func(*args, **kw))
    return impl


