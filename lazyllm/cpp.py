import importlib
import os
import sys
import ctypes
from typing import Dict, Iterable

_LAZYLLM_CPP_MODULE = None
_LAZYLLM_CPP_ENABLED = None


def _is_enabled() -> bool:
    global _LAZYLLM_CPP_ENABLED
    if _LAZYLLM_CPP_ENABLED is None:
        value = os.getenv('LAZYLLM_ENABLE_CPP_OVERRIDE')
        _LAZYLLM_CPP_ENABLED = value is not None and (value == '1' or value.lower() == 'true')
    return _LAZYLLM_CPP_ENABLED


def override_with_cpp_exports(module_globals: Dict[str, object], names: Iterable[str]):
    if not _is_enabled():
        return

    global _LAZYLLM_CPP_MODULE
    if _LAZYLLM_CPP_MODULE is None:
        _LAZYLLM_CPP_MODULE = importlib.import_module('lazyllm.lazyllm_cpp')

    missing = object()
    for name in names:
        cpp_export = getattr(_LAZYLLM_CPP_MODULE, name, missing)
        if cpp_export is missing:
            raise AttributeError(f"module 'lazyllm.lazyllm_cpp' has no attribute '{name}'")
        module_globals[name] = cpp_export
