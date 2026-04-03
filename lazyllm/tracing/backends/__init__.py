from .langfuse import LangfuseBackend


_BACKENDS = {
    LangfuseBackend.name: LangfuseBackend(),
}


def get_tracing_backend(name: str):
    if name not in _BACKENDS:
        raise ValueError(f'Unsupported trace backend: {name}')
    return _BACKENDS[name]
