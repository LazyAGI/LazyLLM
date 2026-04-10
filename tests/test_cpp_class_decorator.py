import importlib.util
import os
import pytest
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class _ConfigStub:
    def __init__(self):
        self._values = {'home': str(Path.home() / '.lazyllm')}

    def add(self, name, _type, default, env_name, *args, **kwargs):
        raw = os.getenv(f'LAZYLLM_{env_name}')
        if raw is None:
            value = default
        elif _type is bool:
            value = raw.lower() in {'1', 'true', 'yes', 'on'}
        else:
            value = _type(raw)
        self._values[name] = value
        setattr(self, name, value)

    def __getitem__(self, key):
        return self._values[key]


def _reload_cpp_module():
    module_path = Path(__file__).resolve().parents[1] / 'lazyllm' / 'cpp.py'
    module_name = 'lazyllm.cpp'

    sys.modules.pop(module_name, None)
    sys.modules.pop('lazyllm', None)

    pkg = types.ModuleType('lazyllm')
    pkg.__path__ = [str(module_path.parent)]  # type: ignore[attr-defined]
    pkg.config = _ConfigStub()  # type: ignore[attr-defined]
    sys.modules['lazyllm'] = pkg

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    cpp = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = cpp
    spec.loader.exec_module(cpp)
    return cpp


def test_cpp_class_keeps_python_class_when_disabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '0')
    cpp = _reload_cpp_module()

    class PyOnly:
        pass

    replaced = cpp.cpp_class(PyOnly)
    assert replaced is PyOnly


def test_cpp_class_replaces_with_cpp_export_when_enabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    class CppDummy:
        pass

    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(Dummy=CppDummy))

    class Dummy:
        pass

    replaced = cpp.cpp_class(Dummy)
    assert replaced is CppDummy


def test_cpp_class_rejects_non_class_object(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    with pytest.raises(TypeError, match='can only decorate classes'):
        cpp.cpp_class('NotAClass')


def test_cpp_class_raises_when_cpp_export_missing(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()
    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace())

    class Missing:
        pass

    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        cpp.cpp_class(Missing)


def test_cpp_class_propagates_import_error_when_enabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    def _boom():
        raise ImportError('boom')

    monkeypatch.setattr(cpp, '_load_cpp_module', _boom)

    class AnyClass:
        pass

    with pytest.raises(ImportError, match='boom'):
        cpp.cpp_class(AnyClass)
