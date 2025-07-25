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
    """一个用于创建可以包含各种项目的流式结构的基类。
这个类提供了一种组织项目的方式，这些项目可以是 ``FlowBase`` 的实例或其他类型，组织成一个层次结构。每个项目都可以有一个名称，结构可以动态地遍历或修改。

Args:
    items (iterable): 要包含在流中的项目的可迭代对象。这些可以是 ``FlowBase`` 的实例或其他对象。
    item_names (list of str, optional): 对应于项目的名称列表。这允许通过名称访问项目。如果未提供，则只能通过索引访问项目。

"""
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

    def __iter__(self):
        # used to support `with pipeline() as (a.b, b):`
        return iter([self, self])

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
        """一个属性，指示当前流项目是否是流结构的根。

**Returns:**

- bool: 如果当前项目没有父级（ ``_father`` 为None），则为True，否则为False。


Examples:
    >>> import lazyllm
    >>> p = lazyllm.pipeline()
    >>> p.is_root
    True
    >>> p2 = lazyllm.pipeline(p)
    >>> p.is_root
    False
    >>> p2.is_root
    True
    """
        return self._father is None

    @property
    def ancestor(self):
        """一个属性，返回当前流项目的最顶层祖先。

如果当前项目是根，则返回其自身。

**Returns:**

- FlowBase: 最顶层的祖先流项目。


Examples:
    >>> import lazyllm
    >>> p = lazyllm.pipeline()
    >>> p2 = lazyllm.pipeline(p)
    >>> p.ancestor is p2
    True
    """
        if self.is_root: return self
        return self._father.ancestor

    def for_each(self, filter, action):
        """对流中每个通过过滤器的项目执行一个操作。

该方法递归地遍历流结构，将操作应用于通过过滤器的每个项目。

Args:
    filter (callable): 一个接受项目作为输入并返回bool的函数，如果该项目应该应用操作，则返回True。
    action (callable): 一个接受项目作为输入并对其执行某些操作的函数。

**Returns:**

- None


Examples:
    >>> import lazyllm
    >>> def test1(): print('1')
    ... 
    >>> def test2(): print('2')
    ... 
    >>> def test3(): print('3')
    ... 
    >>> flow = lazyllm.pipeline(test1, lazyllm.pipeline(test2, test3))
    >>> flow.for_each(lambda x: callable(x), lambda x: print(x))
    <Function type=test1>
    <Function type=test2>
    <Function type=test3>
    """
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
    """一个形成处理阶段管道的顺序执行模型。

 ``Pipeline``类是一个处理阶段的线性序列，其中一个阶段的输出成为下一个阶段的输入。它支持在最后一个阶段之后添加后续操作。它是 ``LazyLLMFlowsBase``的子类，提供了一个延迟执行模型，并允许以延迟方式包装和注册函数。

Args:
    args (list of callables or single callable): 管道的处理阶段。每个元素可以是一个可调用的函数或 ``LazyLLMFlowsBase.FuncWrap``的实例。如果提供了单个列表或元组，则将其解包为管道的阶段。
    post_action (callable, optional): 在管道的最后一个阶段之后执行的可选操作。默认为None。
    kwargs (dict of callables): 管道的命名处理阶段。每个键值对表示一个命名阶段，其中键是名称，值是可调用的阶段。

**Returns:**

- 管道的最后一个阶段的输出。



Examples:
    >>> import lazyllm
    >>> ppl = lazyllm.pipeline(
    ...     stage1=lambda x: x+1,
    ...     stage2=lambda x: f'get {x}'
    ... )
    >>> ppl(1)
    'get 2'
    >>> ppl.stage2
    <Function type=lambda>
    """
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
def barrier(args):
    if _barr.impl: _barr.impl.wait()
    return args


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


config.add('parallel_multiprocessing', bool, False, 'PARALLEL_MULTIPROCESSING')


#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowsBase):
    """用于管理LazyLLMFlows中的并行流的类。

这个类继承自LazyLLMFlowsBase，提供了一个并行或顺序运行操作的接口。它支持使用线程进行并发执行，并允许以字典形式返回结果。


可以这样可视化 ``Parallel`` 类：

```text
#       /> module11 -> ... -> module1N -> out1 \\
# input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#       \> module31 -> ... -> module3N -> out3 /
```        

可以这样可视化 ``Parallel.sequential`` 方法：

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    _scatter (bool, optional): 如果为 ``True``，输入将在项目之间分割。如果为 ``False``，相同的输入将传递给所有项目。默认为 ``False``。
    _concurrent (bool, optional): 如果为 ``True``，操作将使用线程并发执行。如果为 ``False``，操作将顺序执行。默认为 ``True``。
    args: 基类的可变长度参数列表。
    kwargs: 基类的任意关键字参数。

<span style="font-size: 20px;">&ensp;**`asdict property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为dict。当使用 ``asdict`` 时，请务必保证parallel的元素被取了名字，例如:  ``parallel(name=value)`` 。

<span style="font-size: 20px;">&ensp;**`tuple property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为tuple。

<span style="font-size: 20px;">&ensp;**`list property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为list。

<span style="font-size: 20px;">&ensp;**`sum property`**</span>

标记Parellel，使得Parallel每次调用时的返回值做一次累加。

<span style="font-size: 20px;">&ensp;**`join(self, string)`**</span>

标记Parellel，使得Parallel每次调用时的返回值通过 ``string`` 做一次join。


Examples:
    >>> import lazyllm
    >>> test1 = lambda a: a + 1
    >>> test2 = lambda a: a * 4
    >>> test3 = lambda a: a / 2
    >>> ppl = lazyllm.parallel(test1, test2, test3)
    >>> ppl(1)
    (2, 4, 0.5)
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3)
    >>> ppl(1)
    {2, 4, 0.5}
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).asdict
    >>> ppl(2)
    {'a': 3, 'b': 8, 'c': 1.0}
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).astuple
    >>> ppl(-1)
    (0, -4, -0.5)
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).aslist
    >>> ppl(0)
    [1, 0, 0.0]
    >>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\\n')
    >>> ppl(1)
    '2\\n4\\n0.5'
    """

    @staticmethod
    def _worker(func, barrier, global_data, *args, **kw):
        # When multiple threads or processes use the same pipeline, all threads share the same pipeline ID,
        # making it impossible to distinguish between them based on the pipeline ID when saving intermediate
        # results. To address this, we assign a new session ID to each thread to store the intermediate
        # results of each pipeline. Note that when running in parallel, the execution order of modules is not
        # guaranteed, so TODO(wangzhihong) streaming output via FileSystemQueue is not possible.
        lazyllm.globals._init_sid()
        lazyllm.globals._update(global_data)
        lazyllm.globals['bind_args'] = lazyllm.globals['bind_args'].copy()
        _barr.impl = barrier
        r = func(*args, **kw)
        lazyllm.globals.clear()
        return r

    class PostProcessType(Enum):
        NONE = 0
        DICT = 1
        TUPLE = 2
        LIST = 3
        SUM = 4
        JOIN = 5

    def __init__(self, *args, _scatter: bool = False, _concurrent: Union[bool, int] = True,
                 multiprocessing: bool = False, auto_capture: bool = False, **kw):
        super().__init__(*args, **kw, auto_capture=auto_capture)
        self._post_process_type = Parallel.PostProcessType.NONE
        self._post_process_args = None
        self._multiprocessing = multiprocessing or config['parallel_multiprocessing']
        self._concurrent = 0 if not _concurrent else 5 if isinstance(_concurrent, bool) else _concurrent
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
            if self._multiprocessing:
                barrier, executor = None, lazyllm.ProcessPoolExecutor
            else:
                barrier, executor = threading.Barrier(len(items)), concurrent.futures.ThreadPoolExecutor

            with executor(max_workers=self._concurrent) as e:
                futures = [e.submit(partial(self._worker, self.invoke, barrier, lazyllm.globals._data, it, inp, **kw))
                           for it, inp in zip(items, inputs)]
                if (not_done := concurrent.futures.wait(futures).not_done):
                    error_msgs = []
                    for future in not_done:
                        if (exc := future.exception()) is not None:
                            if (tb := getattr(future, "_traceback", None)):
                                tb_str = ''.join(traceback.format_exception(type(exc), exc, tb))
                            else:
                                tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                            error_msgs.append(f"Future: {future}\n{tb_str}")
                        else:
                            error_msgs.append(f"Future: {future} not complete without exception。")
                    raise RuntimeError('Parallel execute failed!\n' + '\n'.join(error_msgs))
                return package([future.result() for future in futures])
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
    """一个流分流器，将输入通过不同的模块以并行方式路由。

Diverter类是一种专门的并行处理形式，其中多个输入分别通过一系列模块并行处理。然后将输出聚合并作为元组返回。

当您拥有可以并行执行的不同数据处理管道，并希望在单个流构造中管理它们时，此类非常有用。

```text
#                 /> in1 -> module11 -> ... -> module1N -> out1 \\
# (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                 \> in3 -> module31 -> ... -> module3N -> out3 /
```                    

Args:
    args: 可变长度参数列表，代表并行执行的模块。
    _concurrent (bool, optional): 控制模块是否应并行执行的标志。默认为 ``True``。可用 ``Diverter.sequential`` 代替 ``Diverter`` 来设置此变量。
    kwargs: 代表额外模块的任意关键字参数，其中键是模块的名称。

.. property:: 
    asdict

    和 ``parallel.asdict`` 一样



Examples:
    >>> import lazyllm
    >>> div = lazyllm.diverter(lambda x: x+1, lambda x: x*2, lambda x: -x)
    >>> div(1, 2, 3)
    (2, 4, -3)
    >>> div = lazyllm.diverter(a=lambda x: x+1, b=lambda x: x*2, c=lambda x: -x).asdict
    >>> div(1, 2, 3)
    {'a': 2, 'b': 4, 'c': -3}
    >>> div(dict(c=3, b=2, a=1))
    {'a': 2, 'b': 4, 'c': -3}
    """
    def __init__(self, *args, _concurrent: Union[bool, int] = True, auto_capture: bool = False, **kw):
        super().__init__(*args, _scatter=True, _concurrent=_concurrent, auto_capture=auto_capture, **kw)


#                  /> in1 \                            /> out1 \
#  (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                  \> in3 /                            \> out3 /
# Attention: Cannot be used in async tasks, ie: training and deploy
# TODO: add check for async tasks
class Warp(Parallel):
    """一个流形变器，将单个模块并行应用于多个输入。

Warp类设计用于将同一个处理模块应用于一组输入。它有效地将单个模块“形变”到输入上，使每个输入都并行处理。输出被收集并作为元组返回。需要注意的是，这个类不能用于异步任务，如训练和部署。

```text
#                 /> in1 \                            /> out1 \
# (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                 \> in3 /                            \> out3 /
```
Args:
    args: 可变长度参数列表，代表要应用于所有输入的单个模块。
    kwargs: 未来扩展的任意关键字参数。

注意:
    - 只允许一个函数在warp中。
    - Warp流不应用于异步任务，如训练和部署。


Examples:
    >>> import lazyllm
    >>> warp = lazyllm.warp(lambda x: x * 2)
    >>> warp(1, 2, 3, 4)
    (2, 4, 6, 8)
    >>> warp = lazyllm.warp(lazyllm.pipeline(lambda x: x * 2, lambda x: f'get {x}'))
    >>> warp(1, 2, 3, 4)
    ('get 2', 'get 4', 'get 6', 'get 8')
    """
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
    """一个根据条件选择并执行流的控制流机制。

 ``Switch``类提供了一种根据表达式的值或条件的真实性选择不同流的方法。它类似于其他编程语言中找到的switch-case语句。

```text
# switch(exp):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
```   

Args:
    args: 可变长度参数列表，交替提供条件和对应的流或函数。条件可以是返回布尔值的可调用对象或与输入表达式进行比较的值。
    post_action (callable, optional): 在执行选定流后要调用的函数。默认为 ``None``。
    judge_on_full_input(bool): 如果设置为 ``True`` ， 则通过 ``switch`` 的输入进行条件判断，否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。
    kwargs: 代表命名条件和对应流或函数的任意关键字参数。

抛出:
    TypeError: 如果提供的参数数量为奇数，或者如果第一个参数不是字典且条件没有成对提供。


Examples:
    >>> import lazyllm
    >>> def is_positive(x): return x > 0
    ...
    >>> def is_negative(x): return x < 0
    ...
    >>> switch = lazyllm.switch(is_positive, lambda x: 2 * x, is_negative, lambda x : -x, 'default', lambda x : '000', judge_on_full_input=True)
    >>>
    >>> switch(1)
    2
    >>> switch(0)
    '000'
    >>> switch(-4)
    4
    >>>
    >>> def is_1(x): return True if x == 1 else False
    ...
    >>> def is_2(x): return True if x == 2 else False
    ...
    >>> def is_3(x): return True if x == 3 else False
    ...
    >>> def t1(x): return 2 * x
    ...
    >>> def t2(x): return 3 * x
    ...
    >>> def t3(x): return x
    ...
    >>> with lazyllm.switch(judge_on_full_input=True) as sw:
    ...     sw.case[is_1::t1]
    ...     sw.case(is_2, t2)
    ...     sw.case[is_3, t3]
    ...
    >>> sw(1)
    2
    >>> sw(2)
    6
    >>> sw(3)
    3
    """
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
    """在LazyLLMFlows框架中实现If-Else功能。

IFS（If-Else Flow Structure）类设计用于根据给定条件的评估有条件地执行两个提供的路径之一（真路径或假路径）。执行选定路径后，可以应用可选的后续操作，并且如果指定，输入可以与输出一起返回。

Args:
    cond (callable): 一个接受输入并返回布尔值的可调用对象。它决定执行哪个路径。如果 ``cond(input)`` 评估为True，则执行 ``tpath`` ；否则，执行 ``fpath`` 。
    tpath (callable): 如果条件为True，则执行的路径。
    fpath (callable): 如果条件为False，则执行的路径。
    post_action (callable, optional): 执行选定路径后执行的可选可调用对象。可以用于进行清理或进一步处理。默认为None。

**Returns:**

- 执行路径的输出。


Examples:
    >>> import lazyllm
    >>> cond = lambda x: x > 0
    >>> tpath = lambda x: x * 2
    >>> fpath = lambda x: -x
    >>> ifs_flow = lazyllm.ifs(cond, tpath, fpath)
    >>> ifs_flow(10)
    20
    >>> ifs_flow(-5)
    5
    """
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
    """初始化一个循环流结构，该结构将一系列函数重复应用于输入，直到满足停止条件或达到指定的迭代次数。

Loop结构允许定义一个简单的控制流，其中一系列步骤在循环中应用，可以使用可选的停止条件来根据步骤的输出退出循环。

Args:
    item (callable or list of callables): 将在循环中应用的函数或可调用对象。
    stop_condition (callable, optional): 一个函数，它接受循环中最后一个项目的输出作为输入并返回一个布尔值。如果返回 ``True``，循环将停止。如果为 ``None``，循环将继续直到达到 ``count``。默认为 ``None``。
    count (int, optional): 运行循环的最大迭代次数。如果为 ``None``，循环将无限期地继续或直到 ``stop_condition`` 返回 ``True``。默认为 ``None``。
    post_action (callable, optional): 循环结束后调用的函数。默认为 ``None``。
    judge_on_full_input(bool): 如果设置为 ``True`` ， 则通过 ``stop_condition`` 的输入进行条件判断，否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

抛出:
    AssertionError: 如果同时提供了 ``stop_condition`` 和 ``count``，或者当提供的 ``count``不是一个整数。


Examples:
    >>> import lazyllm
    >>> loop = lazyllm.loop(lambda x: x * 2, stop_condition=lambda x: x > 10, judge_on_full_input=True)
    >>> loop(1)
    16
    >>> loop(3)
    12
    >>>
    >>> with lazyllm.loop(stop_condition=lambda x: x > 10, judge_on_full_input=True) as lp:
    ...    lp.f1 = lambda x: x + 1
    ...    lp.f2 = lambda x: x * 2
    ...
    >>> lp(0)
    14
    """
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
