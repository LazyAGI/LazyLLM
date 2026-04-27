from importlib import import_module

_SUBMOD_MAP = {
    'collect.context': [
        'LazyTraceContext',
    ],
    'collect.span': [
        'LazySpan',
        'LazyTrace',
    ],
    'collect.runtime': [
        'TracingSetupError',
        'get_trace_context',
        'set_trace_context',
        'tracing_available',
        'enable_trace',
        'current_trace',
        'start_span',
        'set_span_output',
        'set_span_attributes',
        'set_span_error',
        'set_span_usage',
        'finish_span',
    ],
    'collect.configs': [
        'DEFAULT_MODULE_TRACE_CONFIG',
        'get_default_module_trace_config',
        'set_default_module_trace_config',
        'resolve_default_module_trace',
        'resolve_runtime_module_trace_disabled',
    ],
    'collect.hook': [
        'LazyTracingHook',
        'resolve_tracing_hooks',
    ],
    'backends': [
        'get_tracing_backend',
        'get_consume_backend',
    ],
    'backends.base': [
        'TracingBackend',
        'ConsumeBackend',
    ],
    'semantics': [
        'SemanticType',
        'is_valid_trace_id',
        'is_valid_span_id',
        '_TRACE_ID_RE',
        '_SPAN_ID_RE',
        '_VALID_SPAN_KINDS',
        '_VALID_SPAN_STATUS',
    ],
    'consume.api': [
        'get_single_trace',
    ],
    'consume.datamodel.structured': [
        'ExecutionStep',
        'RawData',
        'StructuredTrace',
        'TraceMetadata',
    ],
    'consume.datamodel.raw': [
        'RawSpanRecord',
        'RawTracePayload',
        'RawTraceRecord',
    ],
    'consume.errors': [
        'ConsumeError',
        'TraceNotFound',
        'ConsumeBackendError',
    ],
}
_SUBMOD_MAP_REVERSE = {sym: mod for mod, syms in _SUBMOD_MAP.items() for sym in syms}
__all__ = [sym for syms in _SUBMOD_MAP.values() for sym in syms]


def __getattr__(name: str):
    if name in _SUBMOD_MAP:
        return import_module(f'.{name}', package=__package__)
    if name in _SUBMOD_MAP_REVERSE:
        mod = import_module(f'.{_SUBMOD_MAP_REVERSE[name]}', package=__package__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lazyllm.tracing' has no attribute '{name}'")
