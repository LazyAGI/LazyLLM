import importlib
import os
from typing import TypeVar, cast

_LAZYLLM_CPP_MODULE = None
_LAZYLLM_CPP_ENABLED = None
_C = TypeVar('_C', bound=type)


def _is_enabled() -> bool:
    global _LAZYLLM_CPP_ENABLED
    if _LAZYLLM_CPP_ENABLED is None:
        value = os.getenv('LAZYLLM_ENABLE_CPP_OVERRIDE')
        _LAZYLLM_CPP_ENABLED = value is not None and (value == '1' or value.lower() == 'true')
    return _LAZYLLM_CPP_ENABLED


def _load_cpp_module():
    global _LAZYLLM_CPP_MODULE
    if _LAZYLLM_CPP_MODULE is None:
        _LAZYLLM_CPP_MODULE = importlib.import_module('lazyllm.lazyllm_cpp')
    return _LAZYLLM_CPP_MODULE


def cpp_class(py_class: _C) -> _C:
    if not isinstance(py_class, type):
        raise TypeError(f'@cpp_class can only decorate classes, got: {type(py_class).__name__}')

    if not _is_enabled():
        return py_class

    cpp_module = _load_cpp_module()
    export_name = py_class.__name__

    cpp_export = getattr(cpp_module, export_name)

    return cast(_C, cpp_export)
