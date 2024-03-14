from typing import Any
from lazyllm import LazyLLMRegisterMetaClass, package, bind, root, Thread
import copy
import types
import threading


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
        raise ValueError(f'{self.__class__} object has no attribute {name}')

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

    def __call__(self, args=package()):
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
        return self(*args, **kw)

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


_barr = threading.local()
def barrier(args): _barr.impl.wait(); return args
def _hook(v): _barr.impl = v

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
        nthreads = len(self.items)
        impl = threading.Barrier(nthreads)
        ts = [Thread(target=_impl, args=(it, ), prehook=bind(_hook, impl))
              for it in self.items]
        [t.start() for t in ts]
        return package(t.get_result() for t in ts)


class NamedParallel(Parallel):
    def __init__(self, *, post_action=None, **kw):
        super().__init__(post_action=post_action, **kw)


# parallel in dataflow, serial in executing. dataflow is the same as parallel, while
# it's items will be executed in order. 
class DPES(LazyLLMFlowsBase):
    def _run(self, input=package()):
        def _impl(it):
            try:
                return it(*input) if (isinstance(input, package) and not 
                        isinstance(it, LazyLLMFlowsBase)) else it(input)
            except Exception as e:
                print(f'an error occured when calling {it.__class__.__name__}()')
                raise e
        return package(_impl(it) for it in self.items)


#                  /> in1 -> module11 -> ... -> module1N -> out1 \
#  (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                  \> in3 -> module31 -> ... -> module3N -> out3 /
class Diverter(LazyLLMFlowsBase):
    def _run(self, input=package()):
        assert isinstance(input, package) and len(input) == len(self.items)
        return package(it(inp) for it, inp in zip(self.items, input))


#                  /> in1 \                            /> out1 \
#  (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                  \> in3 /                            \> out3 /
# Attention: Cannot be used in async tasks, ie: training and deploy
# TODO: add check for async tasks
class Warp(LazyLLMFlowsBase):
    def _run(self, input=package()):
        assert isinstance(input, package) and 1 == len(self.items)
        return package(self.items[0](inp) for inp in input)


# switch(exp):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
class Switch(LazyLLMFlowsBase):
    # Switch({cond1: M1, cond2: M2, ..., condN: MN})
    # Switch(cond1, M1, cond2, M2, ..., condN, MN)
    def __init__(self, *args, post_action=None, **kw):
        if len(args) == 1 and isinstance(args[0], dict):
            self.keys, items = list(args[0].keys()), list(args[0].values())
        else:
            self.keys, items = args[0::2], args[1::2]
        super().__init__(*items, post_action=post_action, **kw)

    def _run(self, input=package()):
        assert isinstance(input, package) and len(input) == 2
        exp, input = input
        for idx, cond in enumerate(self.conds):
            if (callable(cond) and cond(exp) is True) or exp == cond:
                return self.items[idx](input) 