try:
    from .langfuse.backend import LangfuseBackend
    _BACKEND_CLASSES = {LangfuseBackend.name: LangfuseBackend}
    _BACKEND_IMPORT_ERRORS = {}
except ImportError as exc:
    _BACKEND_CLASSES = {}
    _BACKEND_IMPORT_ERRORS = {'langfuse': exc}

_BACKEND_INSTANCES = {}


def _unsupported_backend_message(kind: str, name: str, classes: dict, import_errors: dict) -> str:
    available = ', '.join(sorted(classes)) or 'none'
    message = f'Unsupported trace {kind} backend: {name}. Available backends: {available}.'
    if name in import_errors:
        exc = import_errors[name]
        message += f' Backend {name!r} failed to import: {exc.__class__.__name__}: {exc}'
    elif import_errors:
        unavailable = ', '.join(sorted(import_errors))
        message += f' Unavailable backends due to import errors: {unavailable}.'
    return message


def get_tracing_backend(name: str):
    if name not in _BACKEND_CLASSES:
        raise ValueError(_unsupported_backend_message('export', name, _BACKEND_CLASSES, _BACKEND_IMPORT_ERRORS))
    if name not in _BACKEND_INSTANCES:
        _BACKEND_INSTANCES[name] = _BACKEND_CLASSES[name]()
    return _BACKEND_INSTANCES[name]


try:
    from .langfuse import LangfuseConsumeBackend
    _CONSUME_BACKEND_CLASSES = {LangfuseConsumeBackend.name: LangfuseConsumeBackend}
    _CONSUME_BACKEND_IMPORT_ERRORS = {}
except ImportError as exc:
    _CONSUME_BACKEND_CLASSES = {}
    _CONSUME_BACKEND_IMPORT_ERRORS = {'langfuse': exc}

_CONSUME_BACKEND_INSTANCES = {}


def get_consume_backend(name: str):
    if name not in _CONSUME_BACKEND_CLASSES:
        raise ValueError(
            _unsupported_backend_message('consume', name, _CONSUME_BACKEND_CLASSES, _CONSUME_BACKEND_IMPORT_ERRORS)
        )
    if name not in _CONSUME_BACKEND_INSTANCES:
        _CONSUME_BACKEND_INSTANCES[name] = _CONSUME_BACKEND_CLASSES[name]()
    return _CONSUME_BACKEND_INSTANCES[name]


__all__ = [
    'get_consume_backend',
    'get_tracing_backend',
]
