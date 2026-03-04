import importlib
import logging
import os
from typing import Dict, Iterable

LOG = logging.getLogger(__name__)

_CPP_ENABLE_ENV = 'LAZYLLM_ENABLE_CPP_OVERRIDE'


def _is_enabled() -> bool:
    value = os.getenv(_CPP_ENABLE_ENV)
    return value is not None and (value == '1' or value.lower() == 'true')


def _load_cpp_module():
    try:
        return importlib.import_module('lazyllm.lazyllm_cpp')
    except ImportError as e:
        LOG.warning('C++ override is enabled but lazyllm_cpp import failed: %s', e)
        return None


def override_with_cpp_exports(module_globals: Dict[str, object], names: Iterable[str]):
    if not _is_enabled():
        return

    cpp_module = _load_cpp_module()
    if cpp_module is None:
        return

    for name in names:
        if not hasattr(cpp_module, name):
            LOG.error(f'C++ module: {name} does not exist.')
        module_globals[name] = getattr(cpp_module, name)
