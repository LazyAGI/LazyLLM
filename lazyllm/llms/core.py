from lazyllm import LazyLLMRegisterMetaClass
import re

class LLMBase(object, metaclass=LazyLLMRegisterMetaClass):
    pass


reg_template = '''\
class {name}(LazyLLMRegisterMetaClass.all_groups[\'{base}\']):
    pass
'''


def register(cls, templates=reg_template):
    cls = cls.__name__ if isinstance(cls, type) else cls
    cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2].lower() \
        if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls.lower()
    def impl(func):
        func_name = func.__name__
        exec(templates.format(name=func_name, base=cls))
    return impl


