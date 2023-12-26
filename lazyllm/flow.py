from typing import Any
from lazyllm import LazyLLMRegisterMetaClass

class LazyLLMFlowBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *args):
        if isinstance(args[0], (tuple, list)):
            assert len(args) == 1
            args = args[0]
        self.items = args

    def __call__(self, *args, **kw):
        return self.run(*args, **kw)
    
    def run(self, *args, **kw):
        raise NotImplementedError


# input -> module1 -> module2 -> ... -> moduleN -> output
class Pileline(LazyLLMFlowBase):
    def run(self, input):
        output = input
        for it in self.items:
            output = it(output)
        return output

#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowBase):
    def run(self, *inputs, **kw):
        return tuple(it(*inputs, **kw) for it in self.items)
