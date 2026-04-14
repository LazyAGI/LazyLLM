from abc import ABC, abstractmethod
import inspect
import ast
import copy
from typing import Any, Callable, Sequence
from .common import LOG


class LazyLLMHook(ABC):
    __hook_priority__ = 100
    __hook_error_mode__ = 'warn'

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

    def finalize(self):
        raise NotImplementedError


class HookPhaseError(RuntimeError):
    def __init__(self, phase: str, errors: Sequence[tuple[Any, Exception]]):
        self.phase = phase
        self.errors = tuple(errors)
        super().__init__(phase, self.errors)

    def __str__(self):
        names = ', '.join(type(error).__name__ for _, error in self.errors)
        return f'Hook phase `{self.phase}` failed with {len(self.errors)} error(s): {names}'


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


def _materialize_hook(hook_type, obj):
    if isinstance(hook_type, LazyLLMHook):
        return copy.deepcopy(hook_type)
    assert isinstance(hook_type, type) and issubclass(hook_type, LazyLLMHook), (
        f'{hook_type} is not a subclass of LazyLLMHook')
    return hook_type(obj)


def _hook_priority(hook_obj):
    return getattr(hook_obj, '__hook_priority__', 100)


def _hook_error_mode(hook_obj):
    mode = getattr(hook_obj, '__hook_error_mode__', 'warn')
    if mode not in ('warn', 'raise'):
        raise ValueError(f'Invalid hook error mode: {mode}')
    return mode


def _raise_hook_phase_errors(phase: str, errors):
    if not errors:
        return
    raise HookPhaseError(phase, errors)


_builtin_hook_providers: list[Callable[[Any], list]] = []


def register_builtin_hook_provider(provider: Callable[[Any], list]):
    if provider not in _builtin_hook_providers:
        _builtin_hook_providers.append(provider)
    return provider


def resolve_builtin_hooks(obj):
    hooks = []
    for provider in _builtin_hook_providers:
        provided_hooks = provider(obj)
        if provided_hooks:
            hooks.extend(provided_hooks)
    return hooks


def register_hooks(obj, hooks):
    if not hooks:
        return obj
    if not hasattr(obj, '_hooks'):
        raise AttributeError(f'{type(obj).__name__} has no attribute `_hooks`')
    for hook_type in hooks:
        if isinstance(hook_type, LazyLLMHook):
            exists = hook_type in obj._hooks
        else:
            exists = any(h is hook_type for h in obj._hooks if isinstance(h, type))
        if not exists:
            obj._hooks.append(hook_type)
    return obj


def prepare_hooks(obj, hook_types, *args, **kwargs):
    hook_objs = []
    materialized_hooks = []
    for hook_type in hook_types:
        hook_obj = _materialize_hook(hook_type, obj)
        materialized_hooks.append(hook_obj)

    materialized_hooks.sort(key=_hook_priority)

    for hook_obj in materialized_hooks:
        try:
            hook_obj.pre_hook(*args, **kwargs)
        except Exception as e:
            try:
                hook_obj.finalize()
            except Exception as finalize_exc:
                if _hook_error_mode(hook_obj) == 'raise':
                    raise finalize_exc
                LOG.warning(f'Hook `{type(hook_obj).__name__}` finalize failed and will be skipped: {finalize_exc}')
            if _hook_error_mode(hook_obj) == 'raise':
                for active_hook in hook_objs:
                    try:
                        active_hook.finalize()
                    except Exception:
                        pass
                raise
            LOG.warning(f'Hook `{type(hook_obj).__name__}` pre_hook failed and will be skipped: {e}')
            continue
        hook_objs.append(hook_obj)
    return hook_objs


def run_hooks(hook_objs, phase: str, *phase_args):
    if phase not in ('post_hook', 'on_error', 'finalize'):
        raise ValueError(f'Invalid hook phase: {phase}')
    strict_errors = []
    ordered_hooks = hook_objs[::-1]
    for hook_obj in ordered_hooks:
        try:
            getattr(hook_obj, phase)(*phase_args)
        except Exception as e:
            if _hook_error_mode(hook_obj) == 'raise':
                strict_errors.append((hook_obj, e))
            else:
                LOG.warning(f'Hook `{type(hook_obj).__name__}` {phase} failed and will be skipped: {e}')
    _raise_hook_phase_errors(phase, strict_errors)


try:
    from .tracing.hook import resolve_tracing_hooks  # noqa: F401, E402
except ImportError:
    pass
