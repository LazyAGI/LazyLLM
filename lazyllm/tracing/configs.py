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

_module_trace_config = {
    'default': DEFAULT_MODULE_TRACE_CONFIG['default'],
    'by_name': DEFAULT_MODULE_TRACE_CONFIG['by_name'].copy(),
    'by_class': DEFAULT_MODULE_TRACE_CONFIG['by_class'].copy(),
}


def set_default_module_trace_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise TypeError(f'Module trace config must be dict, got {type(config).__name__}')

    _module_trace_config['default'] = config.get('default', DEFAULT_MODULE_TRACE_CONFIG['default'])
    _module_trace_config['by_name'] = dict(config.get('by_name', {}))
    _module_trace_config['by_class'] = dict(config.get('by_class', {}))
    return get_default_module_trace_config()


def get_default_module_trace_config() -> dict:
    return {
        'default': _module_trace_config['default'],
        'by_name': _module_trace_config['by_name'].copy(),
        'by_class': _module_trace_config['by_class'].copy(),
    }


def resolve_default_module_trace(*, module_name=None, module_class=None) -> bool:
    if module_name and module_name in _module_trace_config['by_name']:
        return _module_trace_config['by_name'][module_name]
    if module_class:
        class_names = ([cls.__name__ for cls in module_class.mro()] if isinstance(module_class, type)
                       else [str(module_class)])
        for class_name in class_names:
            if class_name in _module_trace_config['by_class']:
                return _module_trace_config['by_class'][class_name]
    return _module_trace_config['default']
