from ..configs import config
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


config.add('trace_enabled', bool, True, 'TRACE_ENABLED',
           description='Whether LazyLLM tracing is enabled by default.')
config.add('trace_backend', str, 'langfuse', 'TRACE_BACKEND',
           description='The tracing backend used by LazyLLM.')
config.add('trace_content_enabled', bool, True, 'TRACE_CONTENT_ENABLED',
           description='Whether tracing records basic input and output payloads by default.')

__all__ = [
    'TracingBackend',
    'LazyTracingHook',
    'resolve_tracing_hooks',
    'get_tracing_backend',
    'TracingSetupError',
    'start_span',
    'set_span_output',
    'set_span_error',
    'finish_span',
    'get_trace_context',
    'set_trace_context',
    'tracing_available',
    'DEFAULT_MODULE_TRACE_CONFIG',
    'get_default_module_trace_config',
    'set_default_module_trace_config',
    'resolve_default_module_trace',
]
