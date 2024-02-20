from typing import Any
from lazyllm import LazyLLMRegisterMetaClass, package, bind, root
import copy
import types


class FlowBase(object):
    def __init__(self, *items) -> None:
        self._flow_name = None
        self._father = None
        self.items = list(it() if isinstance(it, type) else it for it in items)
        for it in self.items:
            if isinstance(it, FlowBase):
                it._father = self

    def __setattr__(self, name: str, value):
        if isinstance(value, FlowBase) and name != '_father':
            self.items.append(value)
            value._flow_name = name
            value._father = self
        return super().__setattr__(name, value)

    def __getattr__(self, name):
        if 'items' in self.__dict__:
            for it in self.items:
                if getattr(it, '_flow_name', None) == name:
                    return it
        return super(__class__, self).__getattr__(name)

    @property
    def is_root(self):
        return self._father is None

    def for_each(self, filter, action):
        for item in self.items:
            if isinstance(item, FlowBase):
                item.for_each(filter, action)
            elif filter(item):
                action(item)


# TODO(wangzhihong): support workflow launcher.
# Disable item launchers if launcher is already set in workflow.
class LazyLLMFlowsBase(FlowBase, metaclass=LazyLLMRegisterMetaClass):
    class FuncWrap(object):
        def __init__(self, f): self.f = f
        def __call__(self, *args, **kw): return self.f(*args, **kw)
        def __repr__(self): return repr(self.f)
    
    def __init__(self, *args, post_action=None, **kw):
        assert len(args) == 0 or len(kw) == 0
        if len(args) > 0 and isinstance(args[0], (tuple, list)):
            assert len(args) == 1
            args = args[0]
        args = list(args)
        for k, v in kw.items():
            # ensure `_flow_name` is set to object instead of class 
            if isinstance(v, type):
                v = v()
            elif isinstance(v, (types.BuiltinFunctionType, types.FunctionType)):
                # v is copy.deepcopy(v) when v is func, wrap v to set `_flow_name`
                v = LazyLLMFlowsBase.FuncWrap(v)
            else:
                v = v if getattr(v, '_flow_name', None) else copy.deepcopy(v)
            v._flow_name = k
            args.append(v)
        super(__class__, self).__init__(*args)
        self.post_action = post_action() if isinstance(post_action, type) else post_action

    def __call__(self, args):
        output = self._run(args)
        if self.post_action is not None:
            self.post_action(*output) if isinstance(output, package) else self.post_action(output) 
        return output
    
    def _run(self, *args, **kw):
        raise NotImplementedError

    def start(self, *args, **kw):
        assert self.is_root, 'Only root flow can use start()'
        def _exchange(item):
            item._args = [a.get_from(self) if isinstance(a, type(root)) else a for a in item._args]
        self.for_each(lambda x: isinstance(x, bind), _exchange)
        self._run(*args, **kw)

    def __repr__(self):
        representation = '' if self._flow_name is None else (self._flow_name + ' ')
        representation += f'<{self.__class__.__name__}> [\n'
        sub_rep = ',\n'.join([f'{it._flow_name} {it.__repr__()}'
                if (getattr(it, '_flow_name', None) and not isinstance(it, LazyLLMFlowsBase))
                else it.__repr__() for it in self.items])
        sub_rep = '\n'.join(['    ' + s for s in sub_rep.split('\n')])
        representation += sub_rep + '\n]'
        return representation


# input -> module1 -> module2 -> ... -> moduleN -> output
#                                               \> post-action
# TODO(wangzhihong): support mult-input and output
class Pipeline(LazyLLMFlowsBase):
    def _run(self, input=package()):
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
    def __init__(self, *, post_action=None, **kw):
        super().__init__(post_action=post_action, **kw)


#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowsBase):
    def _run(self, input=package()):
        def _impl(it):
            try:
                return it(*input) if (isinstance(input, package) and not 
                        isinstance(it, LazyLLMFlowsBase)) else it(input)
            except Exception as e:
                print(f'an error occured when calling {it.__class__.__name__}()')
                raise e
        return package(_impl(it) for it in self.items)


class NamedParallel(Parallel):
    def __init__(self, *, post_action=None, **kw):
        super().__init__(post_action=post_action, **kw)
