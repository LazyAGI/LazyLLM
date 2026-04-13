from importlib import import_module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # flake8: noqa: E401
    from .runtime import (
        TracingSetupError,
        start_span,
        set_span_output,
        set_span_error,
        finish_span,
        get_trace_context,
        set_trace_context,
        tracing_available,
    )
    from .configs import (
        DEFAULT_MODULE_TRACE_CONFIG,
        get_default_module_trace_config,
        set_default_module_trace_config,
        resolve_default_module_trace,
    )
    from .hook import LazyTracingHook, resolve_tracing_hooks
    from .backends import get_tracing_backend
    from .backends.base import TracingBackend


def __getattr__(name: str):
    if name not in _SUBMOD_MAP_REVERSE and name not in _SUBMOD_MAP:
        raise AttributeError(f"Module 'tracing' has no attribute '{name}'")

    if name in _SUBMOD_MAP:
        return import_module(f'.{name}', package=__package__)
    mod = import_module(f'.{_SUBMOD_MAP_REVERSE[name]}', package=__package__)
    globals()[name] = value = getattr(mod, name)
    return value


_SUBMOD_MAP = {
    'runtime': [
        'TracingSetupError',
        'start_span',
        'set_span_output',
        'set_span_error',
        'finish_span',
        'get_trace_context',
        'set_trace_context',
        'tracing_available',
    ],
    'configs': [
        'DEFAULT_MODULE_TRACE_CONFIG',
        'get_default_module_trace_config',
        'set_default_module_trace_config',
        'resolve_default_module_trace',
    ],
    'hook': [
        'LazyTracingHook',
        'resolve_tracing_hooks',
    ],
    'backends': [
        'get_tracing_backend',
    ],
    'backends.base': [
        'TracingBackend',
    ],
}
_SUBMOD_MAP_REVERSE = {v: k for k, vs in _SUBMOD_MAP.items() for v in vs}
__all__ = sum(_SUBMOD_MAP.values(), [])
