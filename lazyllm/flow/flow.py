import lazyllm
import builtins
from lazyllm import config
from lazyllm.common import LazyLLMRegisterMetaClass, package, kwargs, arguments, bind, root
from lazyllm.common import ReadOnlyWrapper, LOG, globals
from lazyllm.common.bind import _MetaBind
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
from itertools import repeat


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

builtins.isinstance = new_ins

def _is_function(f):
    return isinstance(f, (types.BuiltinFunctionType, types.FunctionType,
                          types.BuiltinMethodType, types.MethodType, types.LambdaType))

class FlowBase(metaclass=_MetaBind):
    """Base class for constructing flow-like structures that can hold multiple items and organize them hierarchically.

This class allows combining different objects (including ``FlowBase`` instances or other types)
into a structured flow, with optional names for each item, enabling both name-based and index-based access.
Items in the structure can be added or traversed dynamically.

Args:
    *items: Items to be included in the flow, which can be instances of ``FlowBase`` or other objects.
    item_names (list of str, optional): A list of names corresponding to the items, paired with ``items`` in order.
        If not provided, all items will be assigned ``None`` as their name.
    auto_capture (bool, optional): Whether to enable automatic variable capture. If ``True``, when used
        as a context manager, newly defined variables in the current scope will be automatically added to the flow.
        Defaults to ``False``.
"""
    def __init__(self, *items, item_names=None, auto_capture=False) -> None:
        self._father = None
        self._items, self._item_names, self._item_ids = [], [], []
        self._auto_capture = auto_capture
        self._capture = True
        self._curr_frame = None
        self._flow_id = str(uuid.uuid4().hex)

        for k, v in zip(item_names if item_names else repeat(None), items):
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
        """Get the identifier for a module or the flow itself. If a string is provided, it is returned as-is. If a bound module is provided, returns its associated item_id. If no argument is given, returns the unique id of the entire flow.

Args:
    module (Optional[Union[str, Any]]): Target module or string identifier.

**Returns:**

- str: Corresponding identifier string.
"""
        if isinstance(module, str): return module
        return self._item_ids[self._items.index(module)] if module else self._flow_id

    @property
    def is_root(self):
        """A property that indicates whether the current flow item is the root of the flow structure.

**Returns:**

- bool: True if the current item has no parent (`` _father`` is None), otherwise False.


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
        """A property that returns the topmost ancestor of the current flow item.

If the current item is the root, it returns itself.

**Returns:**

- FlowBase: The topmost ancestor flow item.


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
        """Performs an action on each item in the flow that matches a given filter.

The method recursively traverses the flow structure, applying the action to each item that passes the filter.

Args:
    filter (callable): A function that takes an item as input and returns True if the item should have the action applied.
    action (callable): A function that takes an item as input and performs some operation on it.

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

bind.__enter__ = _bind_enter
bind.__exit__ = _bind_exit


# TODO(wangzhihong): support workflow launcher.
# Disable item launchers if launcher is already set in workflow.
class LazyLLMFlowsBase(FlowBase, metaclass=LazyLLMRegisterMetaClass):
    """A base class for flow structures with hook support and unified execution logic.

`LazyLLMFlowsBase` is the base class for all LazyLLM flow types. It organizes a sequence of callable modules into a flow and provides support for pre/post hooks, synchronization control, post-processing, and error-safe invocation. It is not intended for direct use but instead serves as a foundational class for concrete flow types like `Pipeline`, `Parallel`, etc.

```text
input --> [Flow module1 -> Flow module2 -> ... -> Flow moduleN] --> output
                   ↑             ↓
               pre_hook       post_hook
```

Args:
    args: A sequence of callables representing the flow modules.
    post_action: An optional callable applied to the output after main flow execution. Defaults to ``None``。
    auto_capture: If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    **kw: Key-value pairs for named components.
"""
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
        """Register a hook type for additional processing before and after the flow execution.

Args:
    hook_type (LazyLLMHook): The hook type or instance to register.
"""
        self._hooks.add(hook_type)

    def unregister_hook(self, hook_type: LazyLLMHook):
        """Unregister a previously registered hook.

Args:
    hook_type (LazyLLMHook): The hook type or instance to remove.
"""
        if hook_type in self._hooks:
            self._hooks.remove(hook_type)

    def clear_hooks(self):
        """Clear all registered hooks.
"""
        self._hooks = set()

    def _post_process(self, output):
        return output

    def _run(self, __input, **kw):
        raise NotImplementedError

    def start(self, *args, **kw):
        """Start flow processing execution (deprecated).

This method is deprecated, it is recommended to directly call the flow instance as a function. Executes the flow processing and returns the result.

Args:
    *args: Variable positional arguments passed to the flow processing.
    **kw: Named arguments passed to the flow processing.

**Returns:**

- The result of flow processing.

**Note:**

- This method is marked as deprecated, please use direct invocation of the flow instance instead.
"""
        lazyllm.LOG.warning('start is depreciated, please use flow as a function instead')
        return self(*args, **kw)

    def set_sync(self, sync=True):
        """Set whether the flow executes synchronously.

Args:
    sync (bool): Whether to execute synchronously. Default is True.

**Returns:**

- LazyLLMFlowsBase: The current instance.
"""
        self._sync = sync
        return self

    def __repr__(self):
        subs = [repr(it) for it in self._items]
        if self.post_action is not None:
            subs.append(lazyllm.make_repr('Flow', 'PostAction', subs=[self.post_action.__repr__()]))
        return lazyllm.make_repr('Flow', self.__class__.__name__, subs=subs, items=self._item_names)

    def wait(self):
        """Wait for all asynchronous tasks in the flow to complete.

**Returns:**

- LazyLLMFlowsBase: The current instance.
"""
        def filter(x):
            return hasattr(x, 'job') and isinstance(x.job, ReadOnlyWrapper) and not x.job.isNone()
        self.for_each(filter, lambda x: x.job.wait())
        return self

    # bind_args: dict(input=input, args=dict(key=value))
    def invoke(self, it, __input, *, bind_args_source=None, **kw):
        """Invoke a target (function, module, or bind object) with the given input.  
Supports root/pipeline output replacement for bind objects.

Args:
    it (Callable | bind): The target to invoke.
    __input (Any): Input data.
    bind_args_source (Any, optional): Source of bind arguments.
    **kw: Additional keyword arguments.
"""
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
        """Bind arguments to the current flow, producing a bind object.

Args:
    *args: Positional arguments.
    **kw: Keyword arguments.

**Returns:**

- bind: The bound bind object.
"""
        return bind(self, *args, **kw)


# input -> module1 -> module2 -> ... -> moduleN -> output
#                                               \> post-action
# TODO(wangzhihong): support mult-input and output
class Pipeline(LazyLLMFlowsBase):
    """A sequential execution model that forms a pipeline of processing stages.

The ``Pipeline`` class is a linear sequence of processing stages, where the output of one stage becomes the input to the next. It supports the addition of post-actions that can be performed after the last stage. It is a subclass of ``LazyLLMFlowsBase`` which provides a lazy execution model and allows for functions to be wrapped and registered in a lazy manner.

Args:
    args (list of callables or single callable): The processing stages of the pipeline. Each element can be a callable function or an instance of ``LazyLLMFlowsBase.FuncWrap``. If a single list or tuple is provided, it is unpacked as the stages of the pipeline.
    post_action (callable, optional): An optional action to perform after the last stage of the pipeline. Defaults to None.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs (dict of callables): Named processing stages of the pipeline. Each key-value pair represents a named stage, where the key is the name and the value is the callable stage.

**Returns:**

- The output of the last stage of the pipeline.


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
    def output(self, module, unpack=False):
        """Get the output result of a specified module in the pipeline.

Args:
    module: The module to get output from. Can be a module object or module name.
    unpack (bool): Whether to unpack the output result. Defaults to False.

**Returns:**

- bind.Args: A bound argument object for data passing in the pipeline.
"""
        return bind.Args(self.id(), self.id(module), unpack=unpack)

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
    """A context manager that temporarily sets whether to save intermediate results during pipeline execution.

When entering the context, `Pipeline.g_save_flow_result` is set to the given value. After exiting, it restores the previous value. Useful for debugging or recording intermediate outputs.

Args:
    flag (bool): Whether to enable result saving. Defaults to True.

**Returns:**

- ContextManager: A context manager.


Examples:
    >>> import lazyllm
    >>> pipe = lazyllm.pipeline(lambda x: x + 1, lambda x: x * 2)
    >>> with lazyllm.save_pipeline_result(True):
    ...     result = pipe(1)
    >>> result
    4
    """
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
    """A class for managing parallel flows in LazyLLMFlows.

This class inherits from LazyLLMFlowsBase and provides an interface for running operations in parallel or sequentially. It supports concurrent execution using threads and allows for the return of results as a dictionary.


The ``Parallel`` class can be visualized as follows:

```text
#       /> module11 -> ... -> module1N -> out1 \\
# input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#       \> module31 -> ... -> module3N -> out3 /
```       

The ``Parallel.sequential`` method can be visualized as follows:

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: Variable length argument list for the base class.
    _scatter (bool, optional): If ``True``, the input is split across the items. If ``False``, the same input is passed to all items. Defaults to ``False``.
    _concurrent (Union[bool, int], optional): If ``True``, operations will be executed concurrently using threading. If an integer, specifies the maximum number of concurrent executions. If ``False``, operations will be executed sequentially. Defaults to ``True``.
    multiprocessing (bool, optional): If ``True``, multiprocessing will be used instead of multithreading for parallel execution. This can provide true parallelism but adds overhead for inter-process communication. Defaults to ``False``.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs: Arbitrary keyword arguments for the base class.

`asdict property`

Tag ``Parallel`` so that the return value of each call to ``Parallel`` is changed from a tuple to a dict. When using ``asdict``, make sure that the elements of ``parallel`` are named, for example: ``parallel(name=value)``.

`astuple property`

Mark Parallel so that the return value of Parallel changes from package to tuple each time it is called.

`aslist property`

Mark Parallel so that the return value of Parallel changes from package to list each time it is called.

`sum property`

Mark Parallel so that the return value of Parallel is accumulated each time it is called.

`join(self, string)`

Mark Parallel so that the return value of Parallel is joined by ``string`` each time it is called.


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
        """Marks the Parallel instance to join its results with the specified string on each call.

Args:
    string (str): The string to use for joining results. Defaults to an empty string.

**Returns:**

- Parallel: Returns the current Parallel instance configured to join results with the specified string.

**Example:**

```python
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\n')
>>> ppl(1)
'2\n4\n0.5'
```
"""
        assert isinstance(string, str), 'argument of join shoule be str'
        return Parallel._set_status(self, type=Parallel.PostProcessType.JOIN, args=string)

    @classmethod
    def sequential(cls, *args, **kw):
        """Creates a Parallel instance that executes sequentially.

This class method sets ``_concurrent`` to ``False``, causing all operations to be executed in sequence rather than in parallel.

The ``Parallel.sequential`` method can be visualized as follows:

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: Variable length argument list passed to the Parallel constructor.
    kwargs: Keyword arguments passed to the Parallel constructor.
    _scatter (bool, optional): If ``True``, the input is split across the items. If ``False``, the same input is passed to all items. Defaults to ``False``.
    _concurrent (bool, optional): If ``True``, operations will be executed concurrently using threading. If ``False``, operations will be executed sequentially. Defaults to ``True``.
    multiprocessing (bool, optional): If ``True``, multiprocessing will be used instead of multithreading for parallel execution. This can provide true parallelism but adds overhead for inter-process communication. Defaults to ``False``.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    args: Variable length argument list for the base class.
    kwargs: Arbitrary keyword arguments for the base class.

**Returns:**

- Parallel: A new Parallel instance configured for sequential execution.
"""
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
                            if (tb := getattr(future, '_traceback', None)):
                                tb_str = ''.join(traceback.format_exception(type(exc), exc, tb))
                            else:
                                tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                            error_msgs.append(f'Future: {future}\n{tb_str}')
                        else:
                            error_msgs.append(f'Future: {future} not complete without exception。')
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
    """A flow diverter that routes inputs through different modules in parallel.

The Diverter class is a specialized form of parallel processing where multiple inputs are each processed by a separate sequence of modules in parallel. The outputs are then aggregated and returned as a tuple.

This class is useful when you have distinct data processing pipelines that can be executed concurrently, and you want to manage them within a single flow construct.

```text
#                 /> in1 -> module11 -> ... -> module1N -> out1 \\
# (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                 \> in3 -> module31 -> ... -> module3N -> out3 /
```                    

Args:
    args : Variable length argument list representing the modules to be executed in parallel.
    _concurrent (bool, optional): A flag to control whether the modules should be run concurrently. Defaults to ``True``. You can use ``Diverter.sequential`` instead of ``Diverter`` to set this variable.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs : Arbitrary keyword arguments representing additional modules, where the key is the name of the module.


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
    """A flow warp that applies a single module to multiple inputs in parallel.

The Warp class is designed to apply the same processing module to a set of inputs. It effectively 'warps' the single module around the inputs so that each input is processed in parallel. The outputs are collected and returned as a tuple. It is important to note that this class cannot be used for asynchronous tasks, such as training and deployment.

```text
#                 /> in1 \                            /> out1 \\
# (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                 \> in3 /                            \> out3 /
``` 

Args:
    args: Variable length argument list representing the single module to be applied to all inputs.
    _scatter (bool): Whether to scatter inputs into parts before processing. Defaults to False.
    _concurrent (bool | int): Whether to execute in parallel. Can be a boolean or a max concurrency limit. Defaults to True.
    auto_capture (bool): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs: Arbitrary keyword arguments for future extensions.

Note:
    - Only one function is allowed in warp.
    - The Warp flow should not be used for asynchronous tasks such as training and deployment.


Examples:
    >>> import lazyllm
    >>> warp = lazyllm.warp(lambda x: x * 2)
    >>> warp(1, 2, 3, 4)
    (2, 4, 6, 8)
    >>> warp = lazyllm.warp(lazyllm.pipeline(lambda x: x * 2, lambda x: f'get {x}'))
    >>> warp(1, 2, 3, 4)
    ('get 2', 'get 4', 'get 6', 'get 8')
    
    >>> from lazyllm import package
    >>> warp1 = lazyllm.warp(lambda x, y: x * 2 + y)
    >>> print(warp1([package(1,2), package(10, 20)]))
    (4, 40)
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
    """A control flow mechanism that selects and executes a flow based on a condition.

The ``Switch`` class provides a way to choose between different flows depending on the value of an expression or the truthiness of conditions. It is similar to a switch-case statement found in other programming languages.

```text
# switch(exp):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
``` 

Args:
    args: A variable length argument list, alternating between conditions and corresponding flows or functions. Conditions are either callables returning a boolean or values to be compared with the input expression.
    conversion (callable, optional): A function used to transform or preprocess the evaluation expression ``exp`` before performing condition matching. Defaults to ``None``.
    post_action (callable, optional): A function to be called on the output after the selected flow is executed. Defaults to ``None``.
    judge_on_full_input(bool): If set to ``True``, the conditional judgment will be performed through the input of ``switch``, otherwise the input will be split into two parts: the judgment condition and the actual input, and only the judgment condition will be judged.

Raises:
    TypeError: If an odd number of arguments are provided, or if the first argument is not a dictionary and the conditions are not provided in pairs.


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

        if self._conversion:
            exp = self._conversion(*exp) if isinstance(exp, package) else self._conversion(exp)

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
    """Implements an If-Else functionality within the LazyLLMFlows framework.

The IFS (If-Else Flow Structure) class is designed to conditionally execute one of two provided
paths (true path or false path) based on the evaluation of a given condition. After the execution
of the selected path, an optional post-action can be applied, and the input can be returned alongside
the output if specified.

Args:
    cond (callable): A callable that takes the input and returns a boolean. It determines which path
                        to execute. If ``cond(input)`` evaluates to True, ``tpath`` is executed; otherwise,
                        ``fpath`` is executed.
    tpath (callable): The path to be executed if the condition is True.
    fpath (callable): The path to be executed if the condition is False.
    post_action (callable, optional): An optional callable that is executed after the selected path.
                                        It can be used to perform cleanup or further processing. Defaults to None.

**Returns:**

- The output of the executed path.


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
    """Initializes a Loop flow structure which repeatedly applies a sequence of functions to an input until a stop condition is met or a specified count of iterations is reached.

The Loop structure allows for the definition of a simple control flow where a series of steps are applied in a loop, with an optional stop condition that can be used to exit the loop early based on the output of the steps.

Args:
    *item (callable or list of callables): The function(s) or callable object(s) that will be applied in the loop.
    stop_condition (callable, optional): A function that takes the output of the last item in the loop as input and returns a boolean. If it returns ``True``, the loop will stop. If ``None``, the loop will continue until ``count`` is reached. Defaults to ``None``.
    count (int, optional): The maximum number of iterations to run the loop for. Defaults to ``sys.maxsize``.
    post_action (callable, optional): A function to be called with the final output after the loop ends. Defaults to ``None``.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    judge_on_full_input (bool): If set to ``True``, the conditional judgment will be performed through the input of ``stop_condition``; otherwise, the input will be split into two parts: the judgment condition and the actual input, and only the judgment condition will be judged.

Raises:
    AssertionError: If the provided ``stop_condition`` is neither callable nor ``None``.


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
    """A complex flow control structure based on Directed Acyclic Graph (DAG).

The Graph class allows you to create complex processing graphs where nodes represent processing functions and edges represent data flow. It supports topological sorting to ensure correct execution order and can handle complex dependencies with multiple inputs and outputs.

The Graph class is particularly suitable for scenarios requiring complex data flow and dependency management, such as machine learning pipelines, data processing workflows, etc.

Args:
    post_action (callable, optional): A function to be called after the graph execution is complete. Defaults to ``None``.
    auto_capture (bool, optional): Whether to automatically capture variables from context. Defaults to ``False``.
    kwargs: Arbitrary keyword arguments representing named nodes and corresponding functions.

**Returns:**

- The final output result of the graph.
"""

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
        """Set the argument names for nodes.

This method is used to set the names of function arguments for nodes in the graph, which is important for correct invocation of multi-parameter functions.

Args:
    arg_names (list): List of argument names, corresponding to the order when nodes were created.


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.add = lambda a, b: a + b
    ...     g.multiply = lambda x, y: x * y
    >>> g.set_node_arg_name([['x', 'y'], ['a', 'b']])
    >>> g._nodes['add'].arg_names
    ['x', 'y']
    >>> g._nodes['multiply'].arg_names
    ['a', 'b']
    """
        for node_name, name in zip(self._item_names, arg_names):
            self._nodes[node_name].arg_names = name

    @property
    def start_node(self):
        """Get the start node of the graph.

**Returns:**

- Node: The start node (__start__) object of the graph.


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.process = lambda x: x * 2
    >>> start = g.start_node
    >>> start.name
    '__start__'
    """
        return self._nodes[Graph.start_node_name]

    @property
    def end_node(self):
        """Get the end node of the graph.

**Returns:**

- Node: The end node (__end__) object of the graph.


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.process = lambda x: x * 2
    >>> end = g.end_node
    >>> end.name
    '__end__'
    """
        return self._nodes[Graph.end_node_name]

    def add_edge(self, from_node, to_node, formatter=None):
        """Add an edge to the graph, defining data flow between nodes.

This method is used to define connection relationships between nodes in the graph, specifying how data flows from one node to another.

Args:
    from_node (str or Node): The name or Node object of the source node.
    to_node (str or Node): The name or Node object of the target node.
    formatter (callable, optional): Optional formatting function for data transformation during transfer. Defaults to ``None``.


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.node1 = lambda x: x * 2
    ...     g.node2 = lambda x: x + 1
    ...     g.node3 = lambda x, y: x + y
    >>> g.add_edge('__start__', 'node1')
    >>> g.add_edge('node1', 'node2')
    >>> g.add_edge('node3', '__end__')
    >>> g._nodes['node1'].outputs
    [<Flow type=Node name=node2>]
    >>> def double_input(data):
    ...     return data * 2
    >>> g.add_edge('node1', 'node3', formatter=double_input)
    >>> g._nodes['node3'].inputs
    {'node1': <function double_input at ...>}
    """
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
        """Add a constant edge that passes a fixed value to a specified node.

This method is used to pass constant values as input to nodes in the graph without needing to get data from other nodes.

Args:
    constant: The constant value to pass.
    to_node (str or Node): The name or Node object of the target node.


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.add = lambda x, y: x + y
    >>> g.add_const_edge(10, 'add')
    >>> g._constants
    [10]
    """
        if isinstance(to_node, (tuple, list)):
            return [self.add_const_edge(constant, t) for t in to_node]
        if isinstance(to_node, str): to_node = self._nodes[to_node]
        to_node.inputs[f'_lazyllm_constant_{len(self._constants)}'] = None
        self._constants.append(constant)

    def topological_sort(self):
        """Perform topological sorting to return the correct node execution order.

This method uses Kahn's algorithm to perform topological sorting on the directed acyclic graph, ensuring all dependencies are satisfied.

**Returns:**

- List[Node]: List of nodes arranged in topological order.

Raises:
- ValueError: If there are circular dependencies in the graph.


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.node1 = lambda x: x * 2
    ...     g.node2 = lambda x: x + 1
    ...     g.node3 = lambda x, y: x + y
    >>> g.add_edge('__start__', 'node1')
    >>> g.add_edge('node1', 'node2')
    >>> g.add_edge('node1', 'node3')
    >>> g.add_edge('node2', 'node3')
    >>> g.add_edge('node3', '__end__')
    >>> sorted_nodes = g.topological_sort()
    >>> [node.name for node in sorted_nodes]
    ['__start__', 'node1', 'node2', 'node3', '__end__']
    >>> g.add_edge('node3', 'node1')
    >>> try:
    ...     g.topological_sort()
    ... except ValueError as e:
    ...     print("检测到循环依赖")
    检测到循环依赖
    """
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
            raise ValueError('Graph has a cycle')

        return [n for n in sorted_nodes if (self._in_degree[n] > 0 or self._out_degree[n] > 0)]

    def _get_input(self, name, node, intermediate_results, futures):
        if name.startswith('_lazyllm_constant_'):
            return self._constants[int(name.removeprefix('_lazyllm_constant_'))]
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

    def compute_node(self, sid, node, intermediate_results, futures):
        """Compute the output result of a single node.

This is an internal method of the graph, used to execute the computation of a single node, including getting input data, applying formatter functions, calling node functions, etc.

Args:
    sid: Session ID.
    node (Node): The node to compute.
    intermediate_results (dict): Intermediate result storage.
    futures (dict): Async task dictionary.

**Returns:**

- The computation result of the node.


Examples:
    >>> import lazyllm
    >>> with lazyllm.graph() as g:
    ...     g.add = lambda x, y: x + y
    ...     g.multiply = lambda x: x * 2
    >>> g.add_edge('__start__', 'add')
    >>> g.add_const_edge(5, 'add')
    >>> g.add_edge('add', 'multiply')
    >>> g.add_edge('multiply', '__end__')
    >>> result = g(3)  # x=3, y=5 (常量)
    >>> result
    16
    """
        globals._init_sid(sid)

        kw = {}
        if len(node.inputs) == 1:
            input = self._get_input(list(node.inputs.keys())[0], node, intermediate_results, futures)
        else:
            # TODO(wangzhihong): add complex rules: mixture of package / support kwargs / ...
            inputs = package(self._get_input(input, node, intermediate_results, futures)
                             for input in node.inputs.keys())
            input = arguments()
            for inp in inputs:
                input.append(inp)

        if isinstance(input, arguments):
            kw = input.kw
            input = input.args

        if node.arg_names:
            if not isinstance(input, (list, tuple, package)): input = [input]
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
