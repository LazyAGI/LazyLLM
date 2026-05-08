from importlib import import_module
from threading import Lock
from typing import Dict, Tuple, Type

_TRACE_BACKEND_SPECS = (('langfuse', '.langfuse.backend', 'LangfuseBackend'),)
_CONSUME_BACKEND_SPECS = (('langfuse', '.langfuse', 'LangfuseConsumeBackend'),)


def _load_backend_classes(specs: Tuple[Tuple[str, str, str], ...]) -> Tuple[Dict[str, Type], Dict[str, Exception]]:
    classes: Dict[str, Type] = {}
    import_errors: Dict[str, Exception] = {}

    for backend_name, module_path, class_name in specs:
        try:
            module = import_module(module_path, package=__package__)
            backend_cls = getattr(module, class_name)
            classes[backend_cls.name] = backend_cls
        except ImportError as exc:
            import_errors[backend_name] = exc

    return classes, import_errors


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


def _get_backend_instance(
    *,
    kind: str,
    name: str,
    classes: Dict[str, Type],
    import_errors: Dict[str, Exception],
    instances: Dict[str, object],
    lock: Lock,
):
    if name not in classes:
        raise ValueError(_unsupported_backend_message(kind, name, classes, import_errors))

    instance = instances.get(name)
    if instance is not None:
        return instance

    with lock:
        instance = instances.get(name)
        if instance is None:
            instance = classes[name]()
            instances[name] = instance
        return instance


_BACKEND_CLASSES, _BACKEND_IMPORT_ERRORS = _load_backend_classes(_TRACE_BACKEND_SPECS)
_CONSUME_BACKEND_CLASSES, _CONSUME_BACKEND_IMPORT_ERRORS = _load_backend_classes(_CONSUME_BACKEND_SPECS)

_BACKEND_INSTANCES: Dict[str, object] = {}
_CONSUME_BACKEND_INSTANCES: Dict[str, object] = {}
_BACKEND_LOCK = Lock()
_CONSUME_BACKEND_LOCK = Lock()


def get_tracing_backend(name: str):
    return _get_backend_instance(
        kind='export',
        name=name,
        classes=_BACKEND_CLASSES,
        import_errors=_BACKEND_IMPORT_ERRORS,
        instances=_BACKEND_INSTANCES,
        lock=_BACKEND_LOCK,
    )


def get_consume_backend(name: str):
    return _get_backend_instance(
        kind='consume',
        name=name,
        classes=_CONSUME_BACKEND_CLASSES,
        import_errors=_CONSUME_BACKEND_IMPORT_ERRORS,
        instances=_CONSUME_BACKEND_INSTANCES,
        lock=_CONSUME_BACKEND_LOCK,
    )


__all__ = ['get_consume_backend', 'get_tracing_backend']
