from typing import Any
from lazyllm import LazyLLMRegisterMetaClass, package


class FlowBase(object):
    def __init__(self, *items) -> None:
        self._flow_name = None
        self.items = list(items)

    def __getattr__(self, name):
        for it in self.items:
            if getattr(it, '_flow_name', None) == name:
                return it
        return super(__class__, self).__getattribute__(name)


# TODO(wangzhihong): support workflow launcher.
# Disable item launchers if launcher is already set in workflow.
class LazyLLMFlowsBase(FlowBase, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *args, post_action=None):
        assert len(args) > 0
        if isinstance(args[0], (tuple, list)):
            assert len(args) == 1
            args = args[0]
        super(__class__, self).__init__(*args)
        self.post_action = post_action

    def __call__(self, args):
        output = self.run(args)
        if self.post_action is not None:
            self.post_action(output)
        return output
    
    def run(self, *args, **kw):
        raise NotImplementedError


# input -> module1 -> module2 -> ... -> moduleN -> output
# TODO(wangzhihong): support mult-input and output
class Pipeline(LazyLLMFlowsBase):
    def run(self, input):
        output = input
        for it in self.items:
            try:
                if isinstance(output, package) and not isinstance(it, LazyLLMFlowsBase):
                    output = it(*output)
                else:
                    output = it(output)
            except Exception as e:
                print(f'an error occured when calling {it.__class__.__name__}()')
                raise e
        return output


class NamedPipeline(Pipeline):
    def __init__(self, **kw):
        post_action = kw.pop('post_action', None)
        args = []
        for k, v in kw.items():
            v._flow_name = k
            args.append(v)
        super().__init__(*args, post_action=post_action)


#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowsBase):
    def run(self, input):
        return tuple(it(*input) if (isinstance(input, package) and not 
                                    isinstance(it, LazyLLMFlowsBase))
                else it(input) for it in self.items)


class NamedParallel(Parallel):
    def __init__(self, **kw):
        post_action = kw.pop('post_action', None)
        args = []
        for k, v in kw.items():
            v._flow_name = k
            args.append(v)
        super().__init__(*args, post_action=post_action)
