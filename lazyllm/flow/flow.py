import lazyllm
import builtins
from lazyllm import LazyLLMRegisterMetaClass, package, kwargs, arguments, bind, root, config
from lazyllm import ReadOnlyWrapper, LOG, globals
from ..common.bind import _MetaBind
from functools import partial
from contextlib import contextmanager
from enum import Enum
import types
import inspect
import threading
import traceback
import sys
import os
from typing import Union, Tuple, List, Optional
import concurrent.futures
from collections import deque
import uuid
from ..hook import LazyLLMHook
import asyncio


class _FuncWrap(object):
    def __init__(self, f):
        self._f = f._f if isinstance(f, _FuncWrap) else f

    def __call__(self, *args, **kw): return self._f(*args, **kw)

    def __repr__(self):
        # TODO: specify lambda/staticmethod/classmethod/instancemethod
        # TODO: add registry message
        return lazyllm.make_repr('Function', (
            self._f if _is_function(self._f) else self._f.__class__).__name__.strip('<>'))

    def __getattr__(self, __key):
        if __key != '_f':
            return getattr(self._f, __key)
        return super(__class__, self).__getattr__(__key)

_oldins = isinstance
def new_ins(obj, cls):
    if _oldins(obj, _FuncWrap) and os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) != 'ON':
        return True if (cls is _FuncWrap or (_oldins(cls, (tuple, list)) and _FuncWrap in cls)) else _oldins(obj._f, cls)
    return _oldins(obj, cls)

setattr(builtins, 'isinstance', new_ins)

def _is_function(f):
    return isinstance(f, (types.BuiltinFunctionType, types.FunctionType,
                          types.BuiltinMethodType, types.MethodType, types.LambdaType))

class FlowBase(metaclass=_MetaBind):
    def __init__(self, *items, item_names=[], auto_capture=False) -> None:
        self._father = None
        self._items, self._item_names, self._item_ids = [], [], []
        self._auto_capture = auto_capture
        self._capture = True
        self._curr_frame = None
        self._flow_id = str(uuid.uuid4().hex)

        for k, v in zip(item_names if item_names else [None] * len(items), items):
            self._add(k, v)

        self._capture = False

    def __post_init__(self): pass

    def _add(self, k, v):
        assert self._capture, f'_add can only be used in `{self.__class__}.__init__` or `with {self.__class__}()`'
        self._items.append(v() if isinstance(v, type) else _FuncWrap(v) if _is_function(v) or v in self._items else v)
        self._item_ids.append(k or str(uuid.uuid4().hex))
        if isinstance(v, FlowBase): v._father = self
        if k:
            assert k not in self._item_names, f'Duplicated names {k}'
            self._item_names.append(k)
        if self._curr_frame and isinstance(v, FlowBase):
            if k:
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
            assert name not in self._item_names, f'Duplicated name: {name}'
            self._add(name, value)
        elif name in getattr(self, '_item_names', ()):
            raise RuntimeError(f'The setting of {self.__class__} elements must be done within the `with` statement.')
        else:
            super(__class__, self).__setattr__(name, value)

    def __getattr__(self, name):
        if '_item_names' in self.__dict__ and name in self._item_names:
            return self._items[self._item_names.index(name)]
        raise AttributeError(f'{self.__class__} object has no attribute {name}')

    def id(self, module=None):
        if isinstance(module, str): return module
        return self._item_ids[self._items.index(module)] if module else self._flow_id

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
        self._hooks = set()

    def __call__(self, *args, **kw):
        hook_objs = []
        for hook_type in self._hooks:
            if isinstance(hook_type, LazyLLMHook):
                hook_objs.append(hook_type)
            else:
                hook_objs.append(hook_type(self))
            hook_objs[-1].pre_hook(*args, **kw)
        output = self._run(args[0] if len(args) == 1 else package(args), **kw)
        if self.post_action is not None: self.invoke(self.post_action, output)
        if self._sync: self.wait()
        r = self._post_process(output)
        for hook_obj in hook_objs[::-1]:
            hook_obj.post_hook(r)
        for hook_obj in hook_objs:
            hook_obj.report()
        return r

    def register_hook(self, hook_type: LazyLLMHook):
        self._hooks.add(hook_type)

    def unregister_hook(self, hook_type: LazyLLMHook):
        if hook_type in self._hooks:
            self._hooks.remove(hook_type)

    def clear_hooks(self):
        self._hooks = set()

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
            if isinstance(self, Pipeline):
                it._args = [self.output(a) if a in self._items else a for a in it._args]
                it._kw = {k: self.output(v) if v in self._items else v for k, v in it._kw.items()}
            kw['_bind_args_source'] = bind_args_source
        try:
            if not isinstance(it, LazyLLMFlowsBase) and isinstance(__input, (package, kwargs)):
                return it(*__input, **kw) if isinstance(__input, package) else it(**__input, **kw)
            else:
                return it(__input, **kw)
        except Exception as e:
            LOG.error(f'An error occored when invoking `{type(it)}({it})` with '
                      f'input {type(__input)}`{__input}` and kw `{kw}`')
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
    g_save_flow_result = None

    def __init__(self, *args, post_action=None, auto_capture=False, **kw):
        super().__init__(*args, post_action=post_action, auto_capture=auto_capture, **kw)
        self.save_flow_result = __class__.g_save_flow_result

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

    @property
    def input(self): return bind.Args(self.id())
    @property
    def kwargs(self): return bind.Args(self.id(), 'kwargs')
    def output(self, module, unpack=False): return bind.Args(self.id(), self.id(module), unpack=unpack)

    def _run(self, __input, **kw):
        output = __input
        bind_args_source = dict(source=self.id(), input=output, kwargs=kw.copy())
        if config['save_flow_result'] or __class__.g_save_flow_result or (
                self.save_flow_result and __class__.g_save_flow_result is not False):
            globals['bind_args'][self.id()] = bind_args_source
        for _ in range(self._loop_count):
            for it in self._items:
                output = self.invoke(it, output, bind_args_source=bind_args_source, **kw)
                kw.clear()
                bind_args_source[self.id(it)] = output
            exp = output
            if not self._judge_on_full_input:
                assert isinstance(output, tuple) and len(output) >= 2
                exp = output[0]
                output = output[1:]
            if callable(self._stop_condition) and self.invoke(self._stop_condition, exp): break
        globals['bind_args'].pop(self.id(), None)
        return output


config.add('save_flow_result', bool, False, 'SAVE_FLOW_RESULT')

@contextmanager
def save_pipeline_result(flag: bool = True):
    old_flag = Pipeline.g_save_flow_result
    Pipeline.g_save_flow_result = flag
    yield
    Pipeline.g_save_flow_result = old_flag


_barr = threading.local()
def barrier(args): _barr.impl.wait(); return args


def _split_input(input: Union[Tuple, List], flag: Optional[Union[int, List]] = None):
    if flag is None or isinstance(flag, int):
        if isinstance(flag, int) and flag > 1 and isinstance(input, package) and len(input) == 1:
            input = input[0]
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

    def __init__(self, *args, _scatter: bool = False, _concurrent: Union[bool, int] = True,
                 auto_capture: bool = False, **kw):
        super().__init__(*args, **kw, auto_capture=auto_capture)
        self._post_process_type = Parallel.PostProcessType.NONE
        self._post_process_args = None

        in_async_env = False
        try:
            in_async_env = asyncio.get_event_loop().is_running()
        except RuntimeError:
            pass
        if in_async_env:
            self._concurrent = 0
        else:
            self._concurrent = _concurrent if not isinstance(_concurrent, bool) else 5 if _concurrent else 0

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

    def join(self, string=''):
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
            def impl(func, barrier, global_data, *args, **kw):
                lazyllm.globals._init_sid()
                lazyllm.globals._update(global_data)
                lazyllm.globals['bind_args'] = lazyllm.globals['bind_args'].copy()
                _barr.impl = barrier
                r = func(*args, **kw)
                lazyllm.globals.clear()
                return r

            loop, barrier = asyncio.new_event_loop(), threading.Barrier(len(items))
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._concurrent) as executor:
                return package(loop.run_until_complete(asyncio.gather(*[loop.run_in_executor(executor, partial(
                    impl, self.invoke, barrier, lazyllm.globals._data, it, inp, **kw))
                    for it, inp in zip(items, inputs)])))
        else:
            return package(self.invoke(it, inp, **kw) for it, inp in zip(items, inputs))

    def _post_process(self, output):
        if self._post_process_type == Parallel.PostProcessType.DICT:
            assert self._item_names, 'Item name should be set when you want to return dict.'
            output = {k: v for k, v in zip(self._item_names, output)}
        elif self._post_process_type == Parallel.PostProcessType.TUPLE:
            output = tuple(output)
        elif self._post_process_type == Parallel.PostProcessType.LIST:
            output = list(output)
        elif self._post_process_type == Parallel.PostProcessType.SUM:
            output = ''.join([str(i) for i in output]) if isinstance(output[0], str) else sum(output, type(output[0])())
        elif self._post_process_type == Parallel.PostProcessType.JOIN:
            output = self._post_process_args.join([str(i) for i in output])
        return output


#                  /> in1 -> module11 -> ... -> module1N -> out1 \
#  (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                  \> in3 -> module31 -> ... -> module3N -> out3 /
class Diverter(Parallel):
    def __init__(self, *args, _concurrent: Union[bool, int] = True, auto_capture: bool = False, **kw):
        super().__init__(*args, _scatter=True, _concurrent=_concurrent, auto_capture=auto_capture, **kw)


#                  /> in1 \                            /> out1 \
#  (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                  \> in3 /                            \> out3 /
# Attention: Cannot be used in async tasks, ie: training and deploy
# TODO: add check for async tasks
class Warp(Parallel):
    def __init__(self, *args, _scatter: bool = False, _concurrent: Union[bool, int] = True,
                 auto_capture: bool = False, **kw):
        super().__init__(*args, _scatter=_scatter, _concurrent=_concurrent, auto_capture=auto_capture, **kw)
        if len(self._items) > 1: self._items = [Pipeline(*self._items)]

    def __post_init__(self):
        if len(self._items) > 1: self._items = [Pipeline(*self._items)]

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
            if (callable(cond) and self.invoke(cond, exp) is True) or (exp == cond) or (
                    exp == package((cond,))) or cond == 'default':
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
        try:
            flag = cond()
        except Exception:
            flag = cond if isinstance(cond, bool) else self.invoke(cond, __input, **kw)
        return self.invoke(tpath if flag else fpath, __input, **kw)


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
        def __init__(self, func, name, arg_names=None):
            self.func, self.name, self.arg_names = func, name, None
            self.inputs, self.outputs = dict(), []

        def __repr__(self): return lazyllm.make_repr('Flow', 'Node', name=self.name)

    def __init__(self, *, post_action=None, auto_capture=False, **kw):
        super(__class__, self).__init__(post_action=post_action, auto_capture=auto_capture, **kw)

    def __post_init__(self):
        self._nodes = {n: Graph.Node(f, n) for f, n in zip(self._items, self._item_names)}
        self._nodes[Graph.start_node_name] = Graph.Node(None, Graph.start_node_name)
        self._nodes[Graph.end_node_name] = Graph.Node(lazyllm.Identity(), Graph.end_node_name)
        self._in_degree = {node: 0 for node in self._nodes.values()}
        self._out_degree = {node: 0 for node in self._nodes.values()}
        self._sorted_nodes = None
        self._constants = []

    def set_node_arg_name(self, arg_names):
        for node_name, name in zip(self._item_names, arg_names):
            self._nodes[node_name].arg_names = name

    @property
    def start_node(self): return self._nodes[Graph.start_node_name]

    @property
    def end_node(self): return self._nodes[Graph.end_node_name]

    def add_edge(self, from_node, to_node, formatter=None):
        if isinstance(from_node, (tuple, list)):
            return [self.add_edge(f, to_node, formatter) for f in from_node]
        if isinstance(to_node, (tuple, list)):
            return [self.add_edge(from_node, t, formatter) for t in to_node]

        if isinstance(from_node, str): from_node = self._nodes[from_node]
        if isinstance(to_node, str): to_node = self._nodes[to_node]
        from_node.outputs.append(to_node)
        assert from_node.name not in to_node.inputs, f'Duplicate edges from {from_node.name} to {to_node.name}'
        to_node.inputs[from_node.name] = formatter
        self._in_degree[to_node] += 1
        self._out_degree[from_node] += 1

    def add_const_edge(self, constant, to_node):
        if isinstance(to_node, (tuple, list)):
            return [self.add_const_edge(constant, t) for t in to_node]
        if isinstance(to_node, str): to_node = self._nodes[to_node]
        to_node.inputs[f'_lazyllm_constant_{len(self._constants)}'] = None
        self._constants.append(constant)

    def topological_sort(self):
        in_degree = self._in_degree.copy()
        queue = deque([node for node in self._nodes.values() if in_degree[node] == 0])
        sorted_nodes: List[Graph.Node] = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for output_node in node.outputs:
                in_degree[output_node] -= 1
                if in_degree[output_node] == 0:
                    queue.append(output_node)

        if len(sorted_nodes) != len(self._nodes):
            raise ValueError("Graph has a cycle")

        return [n for n in sorted_nodes if (self._in_degree[n] > 0 or self._out_degree[n] > 0)]

    def compute_node(self, sid, node, intermediate_results, futures):
        globals._init_sid(sid)

        def get_input(name):
            if name.startswith('_lazyllm_constant_'):
                return self._constants[int(name.strip('_lazyllm_constant_'))]
            if name not in intermediate_results['values']:
                r = futures[name].result()
                with intermediate_results['lock']:
                    if name not in intermediate_results['values']:
                        intermediate_results['values'][name] = r
            r = intermediate_results['values'][name]
            if isinstance(r, Exception): raise r
            if node.inputs[name]:
                if isinstance(r, arguments) and not ((len(r.args) == 0) ^ (len(r.kw) == 0)):
                    raise RuntimeError('Only one of args and kwargs can be given with formatter.')
                r = node.inputs[name]((r.args or r.kw) if isinstance(r, arguments) else r)
            return r

        kw = {}
        if len(node.inputs) == 1:
            input = get_input(list(node.inputs.keys())[0])
        else:
            # TODO(wangzhihong): add complex rules: mixture of package / support kwargs / ...
            inputs = package(get_input(input) for input in node.inputs.keys())
            input = arguments()
            for inp in inputs:
                input.append(inp)

        if isinstance(input, arguments):
            kw = input.kw
            input = input.args

        if node.arg_names:
            kw.update({name: value for name, value in zip(node.arg_names, input)})
            input = package()

        return self.invoke(node.func, input, **kw)

    def _run(self, __input, **kw):
        if not self._sorted_nodes: self._sorted_nodes = self.topological_sort()
        intermediate_results = dict(lock=threading.Lock(), values={})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for node in self._sorted_nodes:
                if node.name == '__start__':
                    intermediate_results['values'][node.name] = (
                        arguments(__input, kw) if (__input and kw) else (kw or __input))
                else:
                    future = executor.submit(self.compute_node, globals._sid, node, intermediate_results, futures)
                    futures[node.name] = future

        return futures[Graph.end_node_name].result()
