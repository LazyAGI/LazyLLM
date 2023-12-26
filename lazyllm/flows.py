from typing import Any
from lazyllm import LazyLLMRegisterMetaClass

# TODO(wangzhihong): support workflow launcher.
# Disable item launchers if launcher is already set in workflow.
class LazyLLMFlowBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *args, post_action=None):
        if isinstance(args[0], (tuple, list)):
            assert len(args) == 1
            args = args[0]
        self.items = args
        self.post_action = post_action

    def __call__(self, *args, **kw):
        output = self.run(*args, **kw)
        if self.post_action is not None:
            self.post_action(output)
        return output
    
    def run(self, *args, **kw):
        raise NotImplementedError


# input -> module1 -> module2 -> ... -> moduleN -> output
# TODO(wangzhihong): support mult-input and output
class Pipeline(LazyLLMFlowBase):
    def run(self, input):
        output = input
        for it in self.items:
            try:
                output = it(output)
            except Exception as e:
                print(f'an error occured when calling {it.__class__.__name__}()')
                raise e
        return output

#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowBase):
    def run(self, *inputs, **kw):
        return tuple(it(*inputs, **kw) for it in self.items)
