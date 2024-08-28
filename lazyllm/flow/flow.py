import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, package, kwargs, arguments, bind, root
from lazyllm import Thread, ReadOnlyWrapper, LOG, globals
from ..common.common import _MetaBind
from functools import partial
from enum import Enum
import types
import inspect
import threading
import traceback
import sys
from typing import Union, Tuple, List, Optional
import concurrent.futures
from collections import deque


class _FuncWrap(object):
    def __init__(self, f):
        self.f = f.f if isinstance(f, _FuncWrap) else f

    def __call__(self, *args, **kw): return self.f(*args, **kw)

    def __repr__(self):
        # TODO: specify lambda/staticmethod/classmethod/instancemethod
        # TODO: add registry message
        return lazyllm.make_repr('Function', self.f.__name__.strip('<>'))

def _is_function(f):
    return isinstance(f, (types.BuiltinFunctionType, types.FunctionType, _FuncWrap,
                          types.BuiltinMethodType, types.MethodType, types.LambdaType))

class FlowBase(metaclass=_MetaBind):
    def __init__(self, *items, item_names=[], auto_capture=False) -> None:
        self._father = None
        self._items, self._item_names = [], []
        self._auto_capture = auto_capture
        self._capture = True
        self._curr_frame = None

        for k, v in zip(item_names if item_names else [None] * len(items), items):
            self._add(k, v)

        self._capture = False

    def __post_init__(self): pass

    def _add(self, k, v):
        assert self._capture, f'_add can only be used in `{self.__class__}.__init__` or `with {self.__class__}()`'
        self._items.append(v() if isinstance(v, type) else _FuncWrap(v) if _is_function(v) else v)
        if isinstance(v, FlowBase): v._father = self
        if k: self._item_names.append(k)
        if self._curr_frame and isinstance(v, FlowBase):
            if k not in self._curr_frame.f_locals:
                self._curr_frame.f_locals[k] = v
            else:
                lazyllm.LOG.warning(f'{k} is already defined in this scope, ignor it')

    def __enter__(self, __frame=None):
        assert len(self._items) == 0, f'Cannot init {self.__class__} with items if you want to use it by context.'
        self._curr_frame = __frame if __frame else inspect.currentframe().f_back
        if self._auto_capture:
            self._frame_keys = list(self._curr_frame.f_locals.keys())
        self._capture = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._auto_capture:
            locals = self._curr_frame.f_locals.copy()
            for var, val in locals.items():
                if var != 'self' and var not in self._frame_keys and (
                        (val._f if isinstance(val, bind) else val) is not self):
                    self._add(var, val)
        self._capture = False
        self._curr_frame = None
        self.__post_init__()
        return False

    def __setattr__(self, name: str, value):
        if '_capture' in self.__dict__ and self._capture and not name.startswith('_'):
            assert name not in self._item_names, 'Duplicated name: {name}'
            self._add(name, value)
        else:
            super(__class__, self).__setattr__(name, value)

    def __getattr__(self, name):
        if '_item_names' in self.__dict__ and name in self._item_names:
            return self._items[self._item_names.index(name)]
        raise AttributeError(f'{self.__class__} object has no attribute {name}')

    @property
    def is_root(self):
        return self._father is None

    @property
    def ancestor(self):
        if self.is_root: return self
        return self._father.ancestor

    def for_each(self, filter, action):
        for item in self._items:
            if isinstance(item, FlowBase):
                item.for_each(filter, action)
            elif filter(item):
                action(item)


def _bind_enter(self):
    assert isinstance(self._f, FlowBase)
    self._f.__enter__(inspect.currentframe().f_back)
    return self

def _bind_exit(self, exc_type, exc_val, exc_tb):
    return self._f.__exit__(exc_type, exc_val, exc_tb)

setattr(bind, '__enter__', _bind_enter)
setattr(bind, '__exit__', _bind_exit)


# TODO(wangzhihong): support workflow launcher.
# Disable item launchers if launcher is already set in workflow.
class LazyLLMFlowsBase(FlowBase, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *args, post_action=None, auto_capture=False, **kw):
        assert len(args) == 0 or len(kw) == 0, f'Cannot provide args `{args}` and kwargs `{kw}` at the same time'
        if len(args) > 0 and isinstance(args[0], (tuple, list)):
            assert len(args) == 1, 'args should be list of callable functions'
            args = args[0]
        args = list(args) + [v() if isinstance(v, type) else v for v in kw.values()]
        super(__class__, self).__init__(*args, item_names=list(kw.keys()), auto_capture=auto_capture)
        self.post_action = post_action() if isinstance(post_action, type) else post_action
        self._sync = False

    def __call__(self, *args, **kw):
        output = self._run(args[0] if len(args) == 1 else package(args), **kw)
        if self.post_action is not None: self.invoke(self.post_action, output)
        if self._sync: self.wait()
        return self._post_process(output)

    def _post_process(self, output):
        return output

    def _run(self, __input, **kw):
        raise NotImplementedError

    def start(self, *args, **kw):
        lazyllm.LOG.warning('start is depreciated, please use flow as a function instead')
        return self(*args, **kw)

    def set_sync(self, sync=True):
        self._sync = sync
        return self

    def __repr__(self):
        subs = [repr(it) for it in self._items]
        if self.post_action is not None:
            subs.append(lazyllm.make_repr('Flow', 'PostAction', subs=[self.post_action.__repr__()]))
        return lazyllm.make_repr('Flow', self.__class__.__name__, subs=subs, items=self._item_names)

    def wait(self):
        def filter(x):
            return hasattr(x, 'job') and isinstance(x.job, ReadOnlyWrapper) and not x.job.isNone()
        self.for_each(filter, lambda x: x.job.wait())
        return self

    # bind_args: dict(input=input, args=dict(key=value))
    def invoke(self, it, __input, *, bind_args_source=None, **kw):
        if isinstance(it, bind):
            if it._has_root:
                it._args = [a.get_from(self.ancestor) if isinstance(a, type(root)) else a for a in it._args]
                it._kw = {k: v.get_from(self.ancestor) if isinstance(v, type(root)) else v for k, v in it._kw.items()}
                it._has_root = False
            if bind_args_source: it = bind(it, _bind_args_source=bind_args_source)
        try:
            if not isinstance(it, LazyLLMFlowsBase) and isinstance(__input, (package, kwargs)):
                return it(*__input, **kw) if isinstance(__input, package) else it(**__input, **kw)
            else:
                return it(__input, **kw)
        except Exception as e:
            LOG.error(f'An error occored when invoking `{type(it)}({it})` with input `{__input}` and kw `{kw}`')
            error_type, error_message = type(e).__name__, str(e)
            tb_str = ''.join(traceback.format_exception(*sys.exc_info()))
            LOG.debug(f'Error type: {error_type}, Error message: {error_message}\n'
                      f'Traceback: {tb_str}')
            raise

    def bind(self, *args, **kw):
        return bind(self, *args, **kw)


# input -> module1 -> module2 -> ... -> moduleN -> output
#                                               \> post-action
# TODO(wangzhihong): support mult-input and output
class Pipeline(LazyLLMFlowsBase):

    @property
    def input(self):
        return bind.Input()

    @property
    def _loop_count(self):
        return getattr(self, '_loop_count_var', 1)

    @_loop_count.setter
    def _loop_count(self, count):
        assert count > 1, 'At least one loop is required!'
        self._loop_count_var = count

    @property
    def _stop_condition(self):
        return getattr(self, '_stop_condition_var', None)

    @_stop_condition.setter
    def _stop_condition(self, cond):
        self._stop_condition_var = cond

    @property
    def _judge_on_full_input(self):
        return getattr(self, '_judge_on_full_input_var', True)

    @_judge_on_full_input.setter
    def _judge_on_full_input(self, judge):
        self._judge_on_full_input_var = judge

    def _run(self, __input, **kw):
        output = __input
        bind_args_source = dict(input=output, args=dict())
        for _ in range(self._loop_count):
            for it in self._items:
                output = self.invoke(it, output, bind_args_source=bind_args_source, **kw)
                kw.clear()
                bind_args_source['args'][id(it)] = output
            exp = output
            if not self._judge_on_full_input:
                assert isinstance(output, tuple) and len(output) >= 2
                exp = output[0]
                output = output[1:]
            if callable(self._stop_condition) and self.invoke(self._stop_condition, exp): break
        return output


_barr = threading.local()
def barrier(args): _barr.impl.wait(); return args
def _hook(v): _barr.impl = v


def _split_input(input: Union[Tuple, List], flag: Optional[Union[int, List]] = None):
    if flag is None or isinstance(flag, int):
        assert isinstance(input, (tuple, list)), (
            f'Only tuple and list input can be split automatically, your input is {input} <{type(input)}>')
        if isinstance(flag, int):
            assert flag == len(input), 'input size mismatch with split number'
        return package(input)
    elif isinstance(flag, list):
        if isinstance(input, dict):
            return package(input[key] for key in flag)
        elif isinstance(input, (tuple, list)):
            return _split_input(input, len(flag))
    raise TypeError(f'invalid flag type {type(flag)} given')


#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowsBase):

    class PostProcessType(Enum):
        NONE = 0
        DICT = 1
        TUPLE = 2
        LIST = 3
        SUM = 4
        JOIN = 5

    def __init__(self, *args, _scatter=False, _concurrent=True, auto_capture=False, **kw):
        super().__init__(*args, **kw, auto_capture=auto_capture)
        self._post_process_type = Parallel.PostProcessType.NONE
        self._post_process_args = None
        self._concurrent = _concurrent
        self._scatter = _scatter

    @staticmethod
    def _set_status(self, type, args=None):
        assert self._post_process_type is Parallel.PostProcessType.NONE, 'Cannor set post process twice'
        self._post_process_type = type
        self._post_process_args = args
        return self

    asdict = property(partial(_set_status, type=PostProcessType.DICT))
    astuple = property(partial(_set_status, type=PostProcessType.TUPLE))
    aslist = property(partial(_set_status, type=PostProcessType.LIST))
    sum = property(partial(_set_status, type=PostProcessType.SUM))

    def join(self, string):
        assert isinstance(string, str), 'argument of join shoule be str'
        return Parallel._set_status(self, type=Parallel.PostProcessType.JOIN, args=string)

    @classmethod
    def sequential(cls, *args, **kw):
        return cls(*args, _concurrent=False, **kw)

    def _run(self, __input, items=None, **kw):
        if items is None:
            items = self._items
            size = len(items)
            if self._scatter:
                inputs = _split_input(__input, self._item_names if self._item_names else size)
            else:
                inputs = [__input] * size
        else:
            inputs = __input

        if self._concurrent:
            nthreads = len(items)
            impl = threading.Barrier(nthreads)
            ts = [Thread(target=self.invoke, args=(it, inp), kwargs=kw, prehook=bind(_hook, impl))
                  for it, inp in zip(items, inputs)]
            [t.start() for t in ts]
            r = package(t.get_result() for t in ts)
        else:
            r = package(self.invoke(it, inp, **kw) for it, inp in zip(items, inputs))
        return r

    def _post_process(self, output):
        if self._post_process_type == Parallel.PostProcessType.DICT:
            assert self._item_names, 'Item name should be set when you want to return dict.'
            output = {k: v for k, v in zip(self._item_names, output)}
        elif self._post_process_type == Parallel.PostProcessType.TUPLE:
            output = tuple(output)
        elif self._post_process_type == Parallel.PostProcessType.LIST:
            output = list(output)
        elif self._post_process_type == Parallel.PostProcessType.SUM:
            output = sum(output, type(output[0])())
        elif self._post_process_type == Parallel.PostProcessType.JOIN:
            output = self._post_process_args.join([str(i) for i in output])
        return output


#                  /> in1 -> module11 -> ... -> module1N -> out1 \
#  (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                  \> in3 -> module31 -> ... -> module3N -> out3 /
class Diverter(Parallel):
    def __init__(self, *args, _concurrent=True, auto_capture=False, **kw):
        super().__init__(*args, _scatter=True, _concurrent=_concurrent, auto_capture=auto_capture, **kw)


#                  /> in1 \                            /> out1 \
#  (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                  \> in3 /                            \> out3 /
# Attention: Cannot be used in async tasks, ie: training and deploy
# TODO: add check for async tasks
class Warp(Parallel):
    def _run(self, __input, **kw):
        assert 1 == len(self._items), 'Only one function is enabled in warp'
        inputs = _split_input(__input)
        items = self._items * len(inputs)
        return super(__class__, self)._run(inputs, items, **kw)

    @property
    def asdict(self): raise NotImplementedError

# switch(conversion(input)):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
class Switch(LazyLLMFlowsBase):
    # Switch({cond1: M1, cond2: M2, ..., condN: MN})
    # Switch(cond1, M1, cond2, M2, ..., condN, MN)
    def __init__(self, *args, conversion=None, post_action=None, judge_on_full_input=True):
        if len(args) == 1 and isinstance(args[0], dict):
            self.conds, items = list(args[0].keys()), list(args[0].values())
        else:
            self.conds, items = list(args[0::2]), args[1::2]
        super().__init__(*items, post_action=post_action)
        self._judge_on_full_input = judge_on_full_input
        self._set_conversion(conversion)

    def _set_conversion(self, conversion):
        self._conversion = conversion

    def _run(self, __input, **kw):
        exp = __input
        if not self._judge_on_full_input:
            assert isinstance(__input, tuple) and len(__input) >= 2
            exp = __input[0]
            __input = __input[1] if len(__input) == 2 else __input[1:]
        if self._conversion: exp = self._conversion(exp)

        for idx, cond in enumerate(self.conds):
            if (callable(cond) and self.invoke(cond, exp) is True) or (exp == cond) or cond == 'default':
                return self.invoke(self._items[idx], __input, **kw)

    class Case:
        def __init__(self, m) -> None: self._m = m
        def __call__(self, cond, func): self._m._add_case(cond, func)

        def __getitem__(self, key):
            if isinstance(key, slice):
                if key.start:
                    if (callable(key.step) and key.stop is None):
                        return self._m._add_case(key.start, key.step)
                    elif (key.step is None and callable(key.stop)):
                        return self._m._add_case(key.start, key.stop)
            elif isinstance(key, tuple) and len(key) == 2 and callable(key[1]):
                return self._m._add_case(key[0], key[1])
            raise RuntimeError(f'Only [cond::func], [cond:func] or [cond, func] is allowed in case, but you give {key}')

    @property
    def case(self): return Switch.Case(self)

    def _add_case(self, case, func):
        self.conds.append(case)
        self._add(None, func)


# result = cond(input) ? tpath(input) : fpath(input)
class IFS(LazyLLMFlowsBase):
    def __init__(self, cond, tpath, fpath, post_action=None):
        super().__init__(cond, tpath, fpath, post_action=post_action)

    def _run(self, __input, **kw):
        cond, tpath, fpath = self._items
        return self.invoke(tpath if self.invoke(cond, __input) else fpath, __input, **kw)


#  in(out) -> module1 -> ... -> moduleN -> exp, out -> out
#      ⬆----------------------------------------|
class Loop(Pipeline):
    def __init__(self, *item, stop_condition=None, count=sys.maxsize, post_action=None,
                 auto_capture=False, judge_on_full_input=True, **kw):
        super().__init__(*item, post_action=post_action, auto_capture=auto_capture, **kw)
        assert callable(stop_condition) or stop_condition is None
        self._judge_on_full_input = judge_on_full_input
        self._stop_condition = stop_condition
        self._loop_count = count


class Graph(LazyLLMFlowsBase):

    start_node_name, end_node_name = '__start__', '__end__'

    class Node:
        def __init__(self, func, name):
            self.func, self.name = func, name
            self.inputs, self.outputs = dict(), []

        def __repr__(self): return lazyllm.make_repr('Flow', 'Node', name=self.name)

    def __init__(self, *, post_action=None, auto_capture=False, **kw):
        super(__class__, self).__init__(post_action=post_action, auto_capture=auto_capture, **kw)

    def __post_init__(self):
        self._nodes = {n: Graph.Node(f, n) for f, n in zip(self._items, self._item_names)}
        self._nodes[Graph.start_node_name] = Graph.Node(None, Graph.start_node_name)
        self._nodes[Graph.end_node_name] = Graph.Node(lambda x: x, Graph.end_node_name)
        self._in_degree = {node: 0 for node in self._nodes.values()}
        self._sorted_nodes = None

    @property
    def start_node(self): return self._nodes[Graph.start_node_name]

    @property
    def end_node(self): return self._nodes[Graph.end_node_name]

    def add_edge(self, from_node, to_node, formatter=None):
        if isinstance(from_node, (tuple, list)):
            for f in from_node: self.add_edge(f, to_node, formatter)
            return
        if isinstance(to_node, (tuple, list)):
            for t in to_node: self.add_edge(from_node, t, formatter)
            return
        if isinstance(from_node, str): from_node = self._nodes[from_node]
        if isinstance(to_node, str): to_node = self._nodes[to_node]
        from_node.outputs.append(to_node)
        assert from_node.name not in to_node.inputs, f'Duplicate edges from {from_node.name} to {to_node.name}'
        to_node.inputs[from_node.name] = formatter
        self._in_degree[to_node] += 1

    def topological_sort(self):
        in_degree = self._in_degree.copy()
        queue = deque([node for node in self._nodes.values() if in_degree[node] == 0])
        sorted_nodes = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for output_node in node.outputs:
                in_degree[output_node] -= 1
                if in_degree[output_node] == 0:
                    queue.append(output_node)

        if len(sorted_nodes) != len(self._nodes):
            raise ValueError("Graph has a cycle")

        return sorted_nodes

    def compute_node(self, sid, node, intermediate_results, futures):
        globals._init_sid(sid)

        def get_input(name):
            if name not in intermediate_results['values']:
                r = futures[name].result()
                with intermediate_results['lock']:
                    if name not in intermediate_results['values']:
                        intermediate_results['values'][name] = r
            r = intermediate_results['values'][name]
            if node.inputs[name]:
                r = node.inputs[name](r)
            return r

        kw = {}
        if len(node.inputs) == 1:
            input = get_input(list(node.inputs.keys())[0])
        else:
            # TODO(wangzhihong): add complex rules: mixture of package / support kwargs / ...
            input = package(get_input(input) for input in node.inputs.keys())
            for inp in input:
                assert not isinstance(inp, (kwargs, package, arguments))

        if isinstance(input, arguments):
            kw = input.kw
            input = input.args
        return self.invoke(node.func, input, **kw)

    def _run(self, __input, **kw):
        if not self._sorted_nodes: self._sorted_nodes = self.topological_sort()
        intermediate_results = dict(lock=threading.Lock(), values={})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for node in self._sorted_nodes:
                if node.name == '__start__':
                    intermediate_results['values'][node.name] = arguments(__input, kw)
                else:
                    future = executor.submit(self.compute_node, globals._sid, node, intermediate_results, futures)
                    futures[node.name] = future

        return futures[Graph.end_node_name].result()
