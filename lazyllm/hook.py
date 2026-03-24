from abc import ABC, abstractmethod
import inspect
import ast
import copy
from .common import LOG
from .tracing.runtime import start_span, set_span_output, set_span_error, finish_span
from .tracing.configs import resolve_default_module_trace

class LazyLLMHook(ABC):

    @abstractmethod
    def __init__(self, obj):
        pass

    @abstractmethod
    def pre_hook(self, *args, **kwargs):
        pass

    @abstractmethod
    def post_hook(self, output):
        pass

    def on_error(self, exc):
        return None

    def report(self):  # This is not an abstract method, but it is required to be implemented in subclasses.
        raise NotImplementedError


def _check_and_get_pre_assign_number(func):
    func_node = ast.parse(inspect.getsource(func)).body[0]

    yield_nodes = [n for n in ast.walk(func_node) if isinstance(n, ast.Yield)]
    yield_count = len(yield_nodes)
    if yield_count == 0: return
    elif yield_count > 1: raise ValueError('function can have at most one yield')

    left_count = 0
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            if any(isinstance(sub, ast.Yield) for sub in ast.walk(node.value)):
                target = node.targets[0]
                left_count = len(target.elts) if isinstance(target, ast.Tuple) else 1
                if left_count > 1: raise ValueError('function can have at most one pre-assign')
                break
    return left_count


class LazyLLMFuncHook(LazyLLMHook):
    def __init__(self, func):
        self._func = func
        self._isgeneratorfunction = inspect.isgeneratorfunction(func)
        if self._isgeneratorfunction:
            self._left_count = _check_and_get_pre_assign_number(func)

    def pre_hook(self, *args, **kwargs):
        if self._isgeneratorfunction:
            self._generator = self._func(*args, **kwargs)
            next(self._generator)
        else:
            self._func(*args, **kwargs)

    def post_hook(self, output):
        assert self._isgeneratorfunction, 'post_hook is only supported for generator functions'
        try:
            self._generator.send(output) if self._left_count == 1 else next(self._generator)
        except StopIteration: pass


class LazyTracingHook(LazyLLMHook):
    def __init__(self, obj):
        self._obj = obj
        self._span_handle = None

    @property
    def _span_kind(self):
        return 'flow' if hasattr(self._obj, '_flow_id') else 'module'

    def _enabled(self) -> bool:
        if self._span_kind != 'module':
            return True
        return resolve_default_module_trace(
            module_name=getattr(self._obj, 'name', None) or getattr(self._obj, '_module_name', None),
            module_class=self._obj.__class__,
        )

    def pre_hook(self, *args, **kwargs):
        if not self._enabled():
            self._span_handle = None
            return
        self._span_handle = start_span(span_kind=self._span_kind, target=self._obj, args=args, kwargs=kwargs)

    def post_hook(self, output):
        set_span_output(self._span_handle, output)

    def on_error(self, exc):
        set_span_error(self._span_handle, exc)

    def report(self):
        finish_span(self._span_handle)
        self._span_handle = None


def _materialize_hook(hook_type, obj):
    if isinstance(hook_type, LazyLLMHook):
        return copy.deepcopy(hook_type)
    assert isinstance(hook_type, type) and issubclass(hook_type, LazyLLMHook), (
        f'{hook_type} is not a subclass of LazyLLMHook')
    return hook_type(obj)


def run_pre_hooks(obj, hook_types, hook_objs, *args, raise_on_error: bool, **kwargs):
    for hook_type in hook_types:
        hook_obj = _materialize_hook(hook_type, obj)
        try:
            hook_obj.pre_hook(*args, **kwargs)
        except Exception as e:
            hook_obj.report()
            if raise_on_error:
                for active_hook in hook_objs:
                    active_hook.report()
                raise
            LOG.warning(f'Hook `{type(hook_obj).__name__}` pre_hook failed and will be skipped: {e}')
            continue
        hook_objs.append(hook_obj)
