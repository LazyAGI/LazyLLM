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


def resolve_runtime_module_trace_disabled(override, *, module_name=None, module_class=None) -> bool:
    """Decide if the runtime override (`globals['trace']['module_trace']`) disables the target.

    Runtime override is single-directional: only explicit False in `by_name` / `by_class`
    turns tracing off. It never re-enables a module that the registry/default has disabled.
    """
    if not isinstance(override, dict):
        return False

    by_name = override.get('by_name')
    if module_name and isinstance(by_name, dict) and by_name.get(module_name) is False:
        return True

    by_class = override.get('by_class')
    if module_class and isinstance(by_class, dict):
        class_names = ([cls.__name__ for cls in module_class.mro()] if isinstance(module_class, type)
                       else [str(module_class)])
        for class_name in class_names:
            if by_class.get(class_name) is False:
                return True

    return False

