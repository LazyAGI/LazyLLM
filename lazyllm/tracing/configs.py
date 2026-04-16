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


def resolve_default_module_trace(*, module_name=None, module_class=None) -> bool:
    with _module_trace_config_lock:
        if module_name and module_name in _module_trace_config['by_name']:
            return _module_trace_config['by_name'][module_name]
        if module_class:
            class_names = ([cls.__name__ for cls in module_class.mro()] if isinstance(module_class, type)
                           else [str(module_class)])
            for class_name in class_names:
                if class_name in _module_trace_config['by_class']:
                    return _module_trace_config['by_class'][class_name]
        return _module_trace_config['default']


def resolve_module_trace(*, module_name=None, module_class=None,
                         runtime_override=None) -> bool:
    """Resolve whether a module should be traced, using a 4-level priority chain:

    1. runtime_override["by_name"][module_name]
    2. DEFAULT_MODULE_TRACE_CONFIG["by_name"]
    3. DEFAULT_MODULE_TRACE_CONFIG["by_class"]
    4. DEFAULT_MODULE_TRACE_CONFIG["default"]
    """
    if runtime_override and isinstance(runtime_override, dict):
        if module_name:
            by_name = runtime_override.get('by_name')
            if isinstance(by_name, dict) and module_name in by_name:
                return bool(by_name[module_name])

    return resolve_default_module_trace(module_name=module_name, module_class=module_class)
