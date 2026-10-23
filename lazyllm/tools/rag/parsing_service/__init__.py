from importlib import import_module

# Importing shared task types must not initialize workers that depend on DocServer.
_EXPORTS = {
    '_Processor': ('.impl', '_Processor'),
    'DocumentProcessor': ('.server', 'DocumentProcessor'),
    'DocumentProcessorWorker': ('.worker', 'DocumentProcessorWorker'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
