import importlib
import os
from typing import List, Optional, TypeVar, cast

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


def _validate_funcs_to_override(funcs_to_override: Optional[List[str]]) -> List[str]:
    if funcs_to_override is None:
        return []
    if not isinstance(funcs_to_override, list):
        raise TypeError(
            f'@cpp_class funcs_to_override must be list[str], got: {type(funcs_to_override).__name__}'
        )
    invalid = [name for name in funcs_to_override if not isinstance(name, str)]
    if invalid:
        raise TypeError('@cpp_class funcs_to_override must be list[str].')
    return funcs_to_override


def cpp_class(py_class: Optional[_C] = None, *, funcs_to_override: Optional[List[str]] = None):
    override_names = _validate_funcs_to_override(funcs_to_override)

    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_class can only decorate classes, got: {type(cls).__name__}')

        if not _is_enabled():
            return cls

        cpp_module = _load_cpp_module()
        export_name = cls.__name__

        cpp_export = getattr(cpp_module, export_name)
        for name in override_names:
            if not hasattr(cls, name):
                raise AttributeError(
                    f'@cpp_class funcs_to_override contains unknown method: {cls.__name__}.{name}'
                )
            method = cls.__dict__.get(name, getattr(cls, name))
            setattr(cpp_export, name, method)
        return cast(_C, cpp_export)

    if py_class is None:
        return _decorate
    return _decorate(py_class)
