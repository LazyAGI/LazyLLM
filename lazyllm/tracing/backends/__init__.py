try:
    from .langfuse.backend import LangfuseBackend
    _BACKEND_CLASSES = {LangfuseBackend.name: LangfuseBackend}
except ImportError:
    _BACKEND_CLASSES = {}

_BACKEND_INSTANCES = {}


def get_tracing_backend(name: str):
    if name not in _BACKEND_CLASSES:
        raise ValueError(f'Unsupported trace backend: {name}')
    if name not in _BACKEND_INSTANCES:
        _BACKEND_INSTANCES[name] = _BACKEND_CLASSES[name]()
    return _BACKEND_INSTANCES[name]


try:
    from .langfuse import LangfuseConsumeBackend
    _CONSUME_BACKEND_CLASSES = {LangfuseConsumeBackend.name: LangfuseConsumeBackend}
except ImportError:
    _CONSUME_BACKEND_CLASSES = {}

_CONSUME_BACKEND_INSTANCES = {}


def get_consume_backend(name: str):
    if name not in _CONSUME_BACKEND_CLASSES:
        raise ValueError(f'Unsupported trace consume backend: {name}')
    if name not in _CONSUME_BACKEND_INSTANCES:
        _CONSUME_BACKEND_INSTANCES[name] = _CONSUME_BACKEND_CLASSES[name]()
    return _CONSUME_BACKEND_INSTANCES[name]


__all__ = [
    'get_consume_backend',
    'get_tracing_backend',
]
