import threading
from ..configs import config

config.add('trace_enabled', bool, True, 'TRACE_ENABLED',
           description='Whether LazyLLM tracing is enabled by default.')
config.add('trace_backend', str, 'langfuse', 'TRACE_BACKEND',
           description='The tracing backend used by LazyLLM.')
config.add('trace_content_enabled', bool, True, 'TRACE_CONTENT_ENABLED',
           description='Whether tracing records basic input and output payloads by default.')


DEFAULT_MODULE_TRACE_CONFIG = {
    'default': True,
    'by_name': {
        'retriever': True,
        'reranker': True,
        'llm': True,
    },
    'by_class': {
        'OnlineModule': True,
    },
}

_module_trace_config_lock = threading.RLock()

_module_trace_config = {
    'default': DEFAULT_MODULE_TRACE_CONFIG['default'],
    'by_name': DEFAULT_MODULE_TRACE_CONFIG['by_name'].copy(),
    'by_class': DEFAULT_MODULE_TRACE_CONFIG['by_class'].copy(),
}


def set_default_module_trace_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise TypeError(f'Module trace config must be dict, got {type(config).__name__}')

    with _module_trace_config_lock:
        _module_trace_config['default'] = config.get('default', DEFAULT_MODULE_TRACE_CONFIG['default'])
        _module_trace_config['by_name'] = dict(config.get('by_name', {}))
        _module_trace_config['by_class'] = dict(config.get('by_class', {}))
    return get_default_module_trace_config()


def get_default_module_trace_config() -> dict:
    with _module_trace_config_lock:
        return {
            'default': _module_trace_config['default'],
            'by_name': _module_trace_config['by_name'].copy(),
            'by_class': _module_trace_config['by_class'].copy(),
        }


_MISSING = object()


def _module_class_names(module_class):
    if module_class is None:
        return ()
    if isinstance(module_class, type):
        return tuple(cls.__name__ for cls in module_class.mro())
    return (str(module_class),)


def _lookup_in_config(cfg, *, module_name, module_class, default=_MISSING):
    if not isinstance(cfg, dict):
        return default

    by_name = cfg.get('by_name')
    if module_name and isinstance(by_name, dict) and module_name in by_name:
        return by_name[module_name]

    by_class = cfg.get('by_class')
    if isinstance(by_class, dict):
        for class_name in _module_class_names(module_class):
            if class_name in by_class:
                return by_class[class_name]

    return default


def resolve_default_module_trace(*, module_name=None, module_class=None) -> bool:
    with _module_trace_config_lock:
        hit = _lookup_in_config(
            _module_trace_config, module_name=module_name, module_class=module_class)
        if hit is not _MISSING:
            return hit
        return _module_trace_config.get('default', DEFAULT_MODULE_TRACE_CONFIG['default'])


def resolve_runtime_module_trace_disabled(override, *, module_name=None, module_class=None) -> bool:
    '''Decide if the runtime override (`globals['trace']['module_trace']`) disables the target.

    Runtime override is single-directional: only explicit False in `by_name` / `by_class`
    turns tracing off. It never re-enables a module that the registry/default has disabled.
    '''
    hit = _lookup_in_config(override, module_name=module_name, module_class=module_class)
    return hit is False
