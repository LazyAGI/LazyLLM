import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, package, kwargs, bind, root
from lazyllm import Thread, ReadOnlyWrapper, LOG
from lazyllm import LazyLlmRequest, ReqResHelper
import types
import threading
import traceback
import sys


class FlowBase(object):
    __enable_request__ = True

    def __init__(self, *items, item_names=None) -> None:
        self._flow_name = None
        self._father = None
        self.items = list(it() if isinstance(it, type) else it for it in items)
        self._item_names = item_names
        for it in self.items:
            if isinstance(it, FlowBase):
                it._father = self

    def __setattr__(self, name: str, value):
        if isinstance(value, FlowBase) and name != '_father':
            self.items.append(value)
            value._father = self
        return super().__setattr__(name, value)

    def __getattr__(self, name):
        if '_item_names' in self.__dict__ and name in self._item_names:
            return self.items[self._item_names.index(name)]
        raise AttributeError(f'{self.__class__} object has no attribute {name}')

    @property
    def is_root(self):
        return self._father is None

    @property
    def ancestor(self):
        if self.is_root: return self
        return self._father.ancestor

    def for_each(self, filter, action):
        for item in self.items:
            if isinstance(item, FlowBase):
                item.for_each(filter, action)
            elif filter(item):
                action(item)


def is_function(f):
    return isinstance(f, (types.BuiltinFunctionType, types.FunctionType,
                          types.BuiltinMethodType, types.MethodType, types.LambdaType))


# TODO(wangzhihong): support workflow launcher.
# Disable item launchers if launcher is already set in workflow.
class LazyLLMFlowsBase(FlowBase, metaclass=LazyLLMRegisterMetaClass):
    class FuncWrap(object):
        def __init__(self, f, name=None):
            self.f = f.f if isinstance(f, LazyLLMFlowsBase.FuncWrap) else f
            self._func_name = name

        def __call__(self, *args, **kw): return self.f(*args, **kw)

        def __repr__(self):
            # TODO: specify lambda/staticmethod/classmethod/instancemethod
            # TODO: add registry message
            return lazyllm.make_repr('Function', self.f.__name__.strip('<>'))

    def __init__(self, *args, post_action=None, return_input=False, **kw):
        assert len(args) == 0 or len(kw) == 0, f'Cannot provide args `{args}` and kwargs `{kw}` at the same time'
        if len(args) > 0 and isinstance(args[0], (tuple, list)):
            assert len(args) == 1, 'args should be list of callable functions'
            args = args[0]
        args = list(args) + [v() if isinstance(v, type) else LazyLLMFlowsBase.FuncWrap(v)
                             if is_function(v) else v for v in kw.values()]
        super(__class__, self).__init__(*args, item_names=list(kw.keys()))
        self.post_action = post_action() if isinstance(post_action, type) else post_action
        self.return_input = return_input
        self.result = None
        self._sync = False

    def __call__(self, *args, **kw):
        helper = ReqResHelper()
        req = helper.make_request(*args, **kw)
        output = helper.make_request(self._run(req))

        if self.post_action is not None: self.invoke(self.post_action, output)
        if self.return_input: output = package(req.input, output.input)
        if self._sync: self.wait()
        return helper.make_response(output)

    def _run(self, input):
        raise NotImplementedError

    def start(self, *args, **kw):
        lazyllm.LOG.warning('start is depreciated, please use flow as a function instead')
        return self(*args, **kw)

    def set_sync(self, sync=True):
        self._sync = sync
        return self

    def __repr__(self):
        subs = [repr(LazyLLMFlowsBase.FuncWrap(it) if is_function(it) else it) for it in self.items]
        if self.post_action is not None:
            subs.append(lazyllm.make_repr('Flow', 'PostAction', subs=[self.post_action.__repr__()]))
        return lazyllm.make_repr('Flow', self.__class__.__name__, subs=subs, items=self._item_names)

    def wait(self):
        def filter(x):
            return hasattr(x, 'job') and isinstance(x.job, ReadOnlyWrapper) and not x.job.isNone()
        self.for_each(filter, lambda x: x.job.wait())
        return self

    def invoke(self, it, input):
        if isinstance(it, bind) and it._has_root:
            it._args = [a.get_from(self.ancestor) if isinstance(a, type(root)) else a for a in it._args]
            it._has_root = False
        try:
            kw = dict()
            if isinstance(input, LazyLlmRequest):
                if getattr(it, '__enable_request__', None):
                    return it(input)
                input, kw = input.input, input.kwargs
            if not isinstance(it, LazyLLMFlowsBase) and isinstance(input, (package, kwargs)):
                return it(*input, **kw) if isinstance(input, package) else it(**input, **kw)
            else:
                return it(input, **kw)
        except Exception as e:
            LOG.error(f'An error occored when invoking `{type(it)}` with `{input}`')
            error_type, error_message = type(e).__name__, str(e)
            tb_str = ''.join(traceback.format_exception(*sys.exc_info()))
            LOG.debug(f'Error type: {error_type}, Error message: {error_message}\n'
                      f'Traceback: {tb_str}')
            raise


# input -> module1 -> module2 -> ... -> moduleN -> output
#                                               \> post-action
# TODO(wangzhihong): support mult-input and output
class Pipeline(LazyLLMFlowsBase):
    def _run(self, input):
        helper = ReqResHelper()
        input = helper.make_request(input)
        for it in self.items:
            input = helper.make_request(self.invoke(it, input))
        return helper.make_response(input)


_barr = threading.local()
def barrier(args): _barr.impl.wait(); return args
def _hook(v): _barr.impl = v

#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowsBase):
    def __init__(self, *args, _scatter=False, _concurrent=True, **kw):
        super().__init__(*args, **kw)
        self._return_dict = False
        self._concurrent = _concurrent
        self._scatter = _scatter

    @property
    def asdict(self):
        self._return_dict = True
        return self

    @classmethod
    def sequential(cls, *args, **kw):
        return cls(*args, _concurrent=False, **kw)

    def _run(self, input, items=None):
        if items is None:
            items = self.items
            size = len(items)
            if self._scatter:
                inputs = input.split(self._item_names if self._item_names else size)
            else:
                inputs = [input] * size
        else:
            inputs = input

        if self._concurrent:
            nthreads = len(items)
            impl = threading.Barrier(nthreads)
            ts = [Thread(target=self.invoke, args=(it, inp), prehook=bind(_hook, impl))
                  for it, inp in zip(items, inputs)]
            [t.start() for t in ts]
            r = package(t.get_result() for t in ts)
        else:
            r = package(self.invoke(it, inp) for it, inp in zip(items, inputs))

        if self._return_dict:
            assert self._item_names, 'Item name should be set when you want to return dict.'
            return {k: v for k, v in zip(self._item_names, r)}
        return r


#                  /> in1 -> module11 -> ... -> module1N -> out1 \
#  (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                  \> in3 -> module31 -> ... -> module3N -> out3 /
class Diverter(Parallel):
    def __init__(self, *args, _concurrent=True, **kw):
        super().__init__(*args, _scatter=True, _concurrent=_concurrent, **kw)


#                  /> in1 \                            /> out1 \
#  (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                  \> in3 /                            \> out3 /
# Attention: Cannot be used in async tasks, ie: training and deploy
# TODO: add check for async tasks
class Warp(Parallel):
    def _run(self, input):
        assert 1 == len(self.items), 'Only one function is enabled in warp'
        inputs = input.split()
        items = self.items * len(inputs)
        return super(__class__, self)._run(inputs, items)

    @property
    def asdict(self): raise NotImplementedError

# switch(exp):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
class Switch(LazyLLMFlowsBase):
    # Switch({cond1: M1, cond2: M2, ..., condN: MN})
    # Switch(cond1, M1, cond2, M2, ..., condN, MN)
    def __init__(self, *args, post_action=None, return_input=False, **kw):
        if len(args) == 1 and isinstance(args[0], dict):
            self.conds, items = list(args[0].keys()), list(args[0].values())
        else:
            self.conds, items = args[0::2], args[1::2]
        items = {repr(k): v for k, v in zip(self.conds, items)}
        super().__init__(**items, post_action=post_action, return_input=return_input, **kw)

    def _run(self, input):
        exp = input
        if isinstance(input.input, package) and len(input.input) == 2 and not callable(self.conds[0]):
            exp = input.input[0]
            input.input = input.input[1]
        for idx, cond in enumerate(self.conds):
            if (callable(cond) and self.invoke(cond, exp) is True) or (exp == cond) or cond == 'default':
                return self.invoke(self.items[idx], input)


# result = cond(input) ? tpath(input) : fpath(input)
class IFS(LazyLLMFlowsBase):
    def __init__(self, cond, tpath, fpath, post_action=None, return_input=False):
        super().__init__(cond, tpath, fpath, post_action=post_action, return_input=return_input)

    def _run(self, input):
        cond, tpath, fpath = self.items
        return self.invoke(tpath, input) if self.invoke(cond, input) else self.invoke(fpath, input)


#  in(out) -> module1 -> ... -> moduleN -> exp, out -> out
#      ⬆----------------------------------------|
class Loop(LazyLLMFlowsBase):
    def __init__(self, *item, stop_condition=None, count=None, post_action=None, return_input=False):
        super().__init__(*item, post_action=post_action, return_input=return_input)
        assert (callable(stop_condition) and count is None) or (
            stop_condition is None and isinstance(count, int))
        self.cond = stop_condition
        self.count = count

    def _run(self, input):
        cnt = 0
        helper = ReqResHelper()
        while True:
            for item in self.items:
                input = helper.make_request(self.invoke(item, input))
            cnt += 1
            if (callable(self.cond) and self.invoke(self.cond, input)) or (self.count is not None and cnt >= self.count):
                break
        return helper.make_response(input)
